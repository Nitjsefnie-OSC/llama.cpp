#!/usr/bin/env python3
"""Affinity tests use a fake Win32 API and never modify a process."""

import argparse
import ctypes
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("bonsai_bench", Path(__file__).with_name("bonsai-server-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class FakeKernel32:
    def __init__(self):
        self.current = 0xFFFFFF
        self.system = 0xFFFFFF
        self.handle = 123
        self.name = r"C:\server\llama-server.exe"
        self.OpenProcess = mock.Mock(return_value=self.handle)
        self.QueryFullProcessImageNameW = mock.Mock(side_effect=self.query_name)
        self.GetProcessAffinityMask = mock.Mock(side_effect=self.get_mask)
        self.SetProcessAffinityMask = mock.Mock(side_effect=self.set_mask)
        self.CloseHandle = mock.Mock(return_value=True)

    def query_name(self, handle, flags, buffer, length):
        assert handle == self.handle
        buffer.value = self.name
        return True

    def get_mask(self, handle, process, system):
        assert handle == self.handle
        ctypes.cast(process, ctypes.POINTER(ctypes.c_size_t)).contents.value = self.current
        ctypes.cast(system, ctypes.POINTER(ctypes.c_size_t)).contents.value = self.system
        return True

    def set_mask(self, handle, mask):
        assert handle == self.handle
        self.current = mask
        return True


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeKernel32()
        patches = [mock.patch.object(bench.sys, "platform", "win32"),
                   mock.patch.object(bench.ctypes, "WinDLL", return_value=self.api, create=True),
                   mock.patch.object(bench.ctypes, "get_last_error", return_value=5, create=True),
                   mock.patch.object(bench.ctypes, "WinError", side_effect=lambda code: OSError(code, "Win32 failure"), create=True),
                   mock.patch.object(bench.sys, "stderr", new_callable=io.StringIO)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def assert_closed(self):
        self.api.CloseHandle.assert_called_once_with(self.api.handle)

    def assert_restored(self):
        self.assertEqual(self.api.current, 0xFFFFFF)
        self.assertEqual(self.api.SetProcessAffinityMask.call_args_list[-1], mock.call(self.api.handle, 0xFFFFFF))
        self.assert_closed()

    def test_decimal_and_hex_masks(self):
        for value in ("65535", "065535", "0xFFFF"):
            self.assertEqual(bench.affinity_mask(value), 65535)
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.affinity_mask(value)

    def test_without_pid_does_not_open_process(self):
        with bench.server_affinity(None, None) as metadata:
            self.assertIsNone(metadata)
        self.api.OpenProcess.assert_not_called()

    def test_change_requires_pid_and_windows(self):
        with self.assertRaises(ValueError):
            with bench.server_affinity(None, 1):
                pass
        with mock.patch.object(bench.sys, "platform", "linux"), self.assertRaises(ValueError):
            with bench.server_affinity(42, 1):
                pass
        self.api.OpenProcess.assert_not_called()

    def test_rejects_32_bit_python(self):
        with mock.patch.object(bench.ctypes, "sizeof", return_value=4), self.assertRaises(RuntimeError):
            with bench.server_affinity(42, 1):
                pass
        self.api.OpenProcess.assert_not_called()

    def test_pid_cannot_wrap_to_another_process(self):
        for pid in (0, -1, 0x10000002A):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                with bench.server_affinity(pid, 1):
                    pass
        self.api.OpenProcess.assert_not_called()

    def test_cli_requires_pid_before_opening_process(self):
        argv = ["bench", "--output", "unused.jsonl", "--affinity-mask", "0xffff"]
        with mock.patch.object(bench.sys, "argv", argv), self.assertRaises(SystemExit) as error:
            bench.main()
        self.assertEqual(error.exception.code, 2)
        self.api.OpenProcess.assert_not_called()

    def test_baseline_inspects_without_setting(self):
        with bench.server_affinity(42, None) as metadata:
            self.assertEqual(metadata, {"original_mask": "0xffffff", "effective_mask": "0xffffff",
                                        "system_mask": "0xffffff", "requested_mask": None})
        self.api.OpenProcess.assert_called_once_with(0x1000, False, 42)
        self.api.SetProcessAffinityMask.assert_not_called()
        self.assert_closed()

    def test_requested_and_restored_masks_are_read_back(self):
        with bench.server_affinity(42, 0xFFFF) as metadata:
            self.assertEqual(self.api.current, 0xFFFF)
            self.assertEqual(metadata["effective_mask"], "0xffff")
        self.api.OpenProcess.assert_called_once_with(0x1200, False, 42)
        self.assertEqual(metadata["restored_mask"], "0xffffff")
        self.assertEqual(self.api.GetProcessAffinityMask.call_count, 3)
        self.assert_restored()

    def test_body_failure_restores(self):
        with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
            with bench.server_affinity(42, 0xFFFF):
                raise RuntimeError("benchmark failed")
        self.assert_restored()

    def test_keyboard_interrupt_restores(self):
        with self.assertRaises(KeyboardInterrupt):
            with bench.server_affinity(42, 0xFFFF):
                raise KeyboardInterrupt()
        self.assert_restored()

    def test_open_failure_does_not_close_invalid_handle(self):
        self.api.OpenProcess.return_value = None
        with self.assertRaises(OSError):
            with bench.server_affinity(42, 1):
                pass
        self.api.CloseHandle.assert_not_called()

    def test_wrong_executable_is_never_modified(self):
        self.api.name = r"C:\other\python.exe"
        with self.assertRaisesRegex(RuntimeError, "not llama-server"):
            with bench.server_affinity(42, 1):
                pass
        self.api.SetProcessAffinityMask.assert_not_called()
        self.assert_closed()

    def test_name_lookup_failure_closes_handle(self):
        self.api.QueryFullProcessImageNameW.side_effect = None
        self.api.QueryFullProcessImageNameW.return_value = False
        with self.assertRaises(OSError):
            with bench.server_affinity(42, 1):
                pass
        self.assert_closed()

    def test_initial_read_failure_closes_handle(self):
        self.api.GetProcessAffinityMask.side_effect = None
        self.api.GetProcessAffinityMask.return_value = False
        with self.assertRaises(OSError):
            with bench.server_affinity(42, 1):
                pass
        self.api.SetProcessAffinityMask.assert_not_called()
        self.assert_closed()

    def test_invalid_subset_is_never_set(self):
        for mask in (0, -1, 1 << 24, 1 << 64):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                with bench.server_affinity(42, mask):
                    pass
        self.api.SetProcessAffinityMask.assert_not_called()
        self.assertEqual(self.api.OpenProcess.call_count, self.api.CloseHandle.call_count)

    def test_initial_set_failure_still_attempts_restore(self):
        def fail_first(handle, mask):
            return False if mask == 0xFFFF else self.api.set_mask(handle, mask)
        self.api.SetProcessAffinityMask.side_effect = fail_first
        with self.assertRaises(OSError):
            with bench.server_affinity(42, 0xFFFF):
                pass
        self.assert_restored()

    def test_requested_readback_mismatch_restores(self):
        def wrong_first(handle, mask):
            return self.api.set_mask(handle, 1 if mask == 0xFFFF else mask)
        self.api.SetProcessAffinityMask.side_effect = wrong_first
        with self.assertRaisesRegex(RuntimeError, "Requested affinity"):
            with bench.server_affinity(42, 0xFFFF):
                pass
        self.assert_restored()

    def test_requested_readback_failure_restores(self):
        def fail_second(handle, process, system):
            if self.api.GetProcessAffinityMask.call_count == 2:
                return False
            return self.api.get_mask(handle, process, system)
        self.api.GetProcessAffinityMask.side_effect = fail_second
        with self.assertRaises(OSError):
            with bench.server_affinity(42, 0xFFFF):
                pass
        self.assert_restored()

    def test_restore_set_failure_is_loud_and_closes(self):
        def fail_restore(handle, mask):
            return False if mask == 0xFFFFFF else self.api.set_mask(handle, mask)
        self.api.SetProcessAffinityMask.side_effect = fail_restore
        with self.assertRaisesRegex(RuntimeError, "Failed to restore"):
            with bench.server_affinity(42, 0xFFFF):
                raise ValueError("benchmark failure must not hide restoration failure")
        self.assert_closed()

    def test_restore_readback_failure_is_loud_and_closes(self):
        def fail_third(handle, process, system):
            return False if self.api.GetProcessAffinityMask.call_count == 3 else self.api.get_mask(handle, process, system)
        self.api.GetProcessAffinityMask.side_effect = fail_third
        with self.assertRaisesRegex(RuntimeError, "Failed to restore"):
            with bench.server_affinity(42, 0xFFFF):
                pass
        self.assert_closed()

    def test_restore_readback_mismatch_is_loud_and_closes(self):
        self.api.SetProcessAffinityMask.side_effect = lambda handle, mask: self.api.set_mask(handle, 0xFFFF)
        with self.assertRaisesRegex(RuntimeError, "Failed to restore"):
            with bench.server_affinity(42, 0xFFFF):
                pass
        self.assert_closed()

    def test_zero_group_masks_are_rejected(self):
        self.api.current = 0
        with self.assertRaisesRegex(RuntimeError, "processor groups"):
            with bench.server_affinity(42, 1):
                pass
        self.api.SetProcessAffinityMask.assert_not_called()
        self.assert_closed()

    def fake_response(self, request, timeout):
        route = request.full_url.rsplit("/", 1)[-1]
        payloads = {"props": {"default_generation_settings": {"n_ctx": 4096}},
                    "tokenize": {"tokens": [1, 2, 3]}, "slots": [{"is_processing": False}],
                    "completion": {"content": "test", "timings": {"prompt_n": 1, "predicted_n": 1,
                                   "prompt_per_second": 400, "predicted_per_second": 30}}}
        return io.StringIO(json.dumps(payloads[route]))

    def run_cli(self, output, changed=True):
        argv = ["bench", "--pid", "42", "--output", str(output), "--prompts", "1", "--tokens", "1", "--reps", "1"]
        if changed:
            argv += ["--affinity-mask", "0xffff"]
        with mock.patch.object(bench.sys, "argv", argv), \
             mock.patch.object(bench.sys, "stdout", new_callable=io.StringIO), \
             mock.patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="gpu counters")), \
             mock.patch.object(bench.subprocess, "check_output", return_value="[]"):
            bench.main()

    def test_cli_baseline_records_affinity_without_change(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bench.urllib.request, "urlopen", side_effect=self.fake_response):
            output = Path(directory) / "baseline.jsonl"
            self.run_cli(output, changed=False)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["affinity"]["effective_mask"], "0xffffff")
            self.assertEqual([row["warmup"] for row in rows if row["kind"] == "trial"], [True, False])
        self.api.SetProcessAffinityMask.assert_not_called()
        self.assert_closed()

    def test_cli_changed_run_records_effective_mask(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bench.urllib.request, "urlopen", side_effect=self.fake_response):
            output = Path(directory) / "changed.jsonl"
            self.run_cli(output)
            metadata = json.loads(output.read_text().splitlines()[0])
            self.assertEqual(metadata["affinity"]["original_mask"], "0xffffff")
            self.assertEqual(metadata["affinity"]["effective_mask"], "0xffff")
            self.assertEqual(metadata["affinity"]["system_mask"], "0xffffff")
        self.assert_restored()

    def test_cli_api_failure_restores(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(bench.urllib.request, "urlopen", side_effect=OSError("API failed")), \
             self.assertRaisesRegex(OSError, "API failed"):
            self.run_cli(Path(directory) / "failed.jsonl")
        self.assert_restored()

    def test_cli_existing_output_restores_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bench.urllib.request, "urlopen", side_effect=self.fake_response):
            output = Path(directory) / "existing.jsonl"
            output.write_text("keep")
            with self.assertRaises(FileExistsError):
                self.run_cli(output)
            self.assertEqual(output.read_text(), "keep")
        self.assert_restored()

    def test_cli_output_write_failure_restores(self):
        class BrokenOutput(io.StringIO):
            def write(self, value):
                raise OSError("output write failed")
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(bench.urllib.request, "urlopen", side_effect=self.fake_response), \
             mock.patch.object(bench.Path, "open", return_value=BrokenOutput()), \
             self.assertRaisesRegex(OSError, "output write failed"):
            self.run_cli(Path(directory) / "failed.jsonl")
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
