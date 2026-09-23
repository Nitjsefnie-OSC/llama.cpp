#!/usr/bin/env python3
"""Synthetic host trace fixtures only; never service or CUDA execution."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('bonsai-host-phases-check.py')
SPEC = importlib.util.spec_from_file_location('host_check', SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
CLI_RUNS = []


def run_cli(args):
    r = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    CLI_RUNS.append(dict(argv=args[:], exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr))
    return r


def span(i, parent, phase, reason, start, end, inherit=None, **kw):
    r = dict(v=1, event='span', pid=10, tid=20, thread=1, batch=1, id=i, parent=parent,
             phase=phase, reason=reason, start_us=start, end_us=end, ctx='0x100', lifetime=2,
             attempt=1, ubatch=0, tokens=1, outputs=1, graph='0x0', uid=0, key='0x0',
             backend='0x0', device=-1, mode='unbound', invocation=0, bytes=0,
             direction='none', stream='unknown', result_known=0, result=0, unwind=0)
    if inherit:
        for k in CHECK.BINDING: r[k] = inherit[k]
    r.update(kw)
    return r


def batch(spans, batch_id=1, prior_used=0, flush=(101, 110)):
    common = {k:spans[0][k] for k in ('v', 'pid', 'tid', 'thread')}
    common['batch'] = batch_id
    for s in spans: s.update(common)
    groups = {}
    for s in spans:
        key = (s['phase'], s['reason'], s['direction'])
        g = groups.setdefault(key, dict(**common, event='aggregate', phase=key[0], reason=key[1],
                                       direction=key[2], count=0, bytes=0, sum_us=0, max_us=0))
        duration = s['end_us'] - s['start_us']
        g['count'] += 1; g['bytes'] += s['bytes']; g['sum_us'] += duration
        g['max_us'] = max(g['max_us'], duration)
    return [dict(**common, event='batch', clock='ggml_time_us', unit='us', records=len(spans),
                 dropped=0, budget=262144, used=prior_used+len(spans), valid=1, flush_start_us=flush[0])
            ] + spans + list(groups.values()) + [dict(**common, event='end', records=len(spans),
               aggregates=len(groups), dropped=0, valid=1, flush_end_us=flush[1])]


def fixture(mode='replay', split_uid=0):
    d = span(1, 0, 'decode', 'llama_context_decode', 10, 100, result_known=1)
    u = span(2, 1, 'ubatch', 'decode_ubatch', 20, 90, inherit=d, ubatch=1)
    s = span(3, 2, 'scheduler_enqueue', 'ggml_backend_sched_graph_compute_async', 30, 65,
             inherit=u, graph='0x200', uid=9)
    c = span(4, 3, 'cuda_graph_compute', 'ggml_backend_cuda_graph_compute', 35, 60, inherit=s,
             graph='0x300', uid=split_uid, key='0x400', backend='0x500', device=0, mode=mode, invocation=3)
    submit = span(5, 4, 'cuda_graph_submit', 'selected_graph_mode', 40, 59, inherit=c)
    launch = span(6, 5, 'cuda_graph_launch', 'cudaGraphLaunch', 45, 50, inherit=submit)
    l = span(7, 2, 'logits_enqueue', 'logits_d2h', 70, 85, inherit=u, bytes=993280, direction='d2h')
    a = span(8, 7, 'cuda_async_copy', 'ggml_backend_cuda_get_tensor_async', 75, 80, inherit=l,
             bytes=993280, direction='d2h', backend='0x500', device=0, stream='backend_existing')
    first = batch([d,u,s,c,submit,launch,l,a])
    g = span(9, 0, 'output_getter', 'llama_get_logits_ith', 120, 155, tokens=-1)
    x = span(10, 9, 'context_sync', 'synchronize', 125, 150, inherit=g)
    y = span(11, 10, 'scheduler_sync', 'ggml_backend_sched_synchronize', 130, 148, inherit=x)
    z = span(12, 11, 'cuda_backend_sync', 'ggml_backend_cuda_synchronize', 135, 140, inherit=y,
             backend='0x500', device=0, stream='backend_existing')
    return first + batch([g,x,y,z], 2, 8, (156,160))


def wire(records):
    return ['SYNTHETIC FIXTURE; NOT AN ACTUAL SERVICE CAPTURE'] + [
        '[prefix] CUDA_HOST_PHASES,'+','.join(f'{k}={v}' for k,v in r.items()) for r in records]


def pick(records, i):
    return next(r for r in records if r['event']=='span' and r['id']==i)


def request_fixture(phase, children, size=4, unwind=0):
    """Synthetic one API request with explicitly observed immediate callbacks."""
    reason = {'logits_enqueue':'logits_d2h', 'tensor_upload':'ggml_backend_tensor_set',
              'tensor_upload_async':'ggml_backend_tensor_set_async'}[phase]
    root = span(1,0,phase,reason,10,90,ctx='0x0',lifetime=0,attempt=0,tokens=-1,
                outputs=-1,bytes=size,direction='d2h' if phase=='logits_enqueue' else 'none',unwind=unwind)
    rows=[root]
    for i,(p,reason,direction,nbytes) in enumerate(children,2):
        rows.append(span(i,1,p,reason,10+i*5,14+i*5,inherit=root,bytes=nbytes,
                         direction=direction,backend='0x500',device=0,
                         stream='backend_existing' if p in CHECK.CUDA_IO-{'cuda_buffer_copy','cuda_buffer_sync'} else 'per_thread'))
    return batch(rows)


def decode_fixture(tokens=6, outputs=2, result=0, parts=((4,1),(2,1)), unwind=0):
    root=span(1,0,'decode','llama_context_decode',10,100,tokens=tokens,outputs=outputs,
              result_known=int(not unwind),result=result,unwind=unwind)
    rows=[root]
    for i,(nt,no) in enumerate(parts,2):
        rows.append(span(i,1,'ubatch','decode_ubatch',i*10,i*10+5,inherit=root,
                         ubatch=i-1,tokens=nt,outputs=no))
    return batch(rows)


class HostTests(unittest.TestCase):
    def result(self, records): return CHECK.analyze(wire(records))
    def fails(self, records):
        with self.assertRaises(ValueError): self.result(records)

    def test_valid_split_graph_uid_zero(self):
        r=self.result(fixture());self.assertTrue(r['structurally_valid'])
        self.assertFalse(r['external_coverage']['complete']);self.assertEqual(r['spans'],12)
    def test_capture(self): self.assertTrue(self.result(fixture('capture'))['structurally_valid'])
    def test_direct_has_no_launch(self):
        r=fixture('direct');ss=[x for x in r if x['event']=='span' and x['batch']==1 and x['id']!=6]
        for x in ss:
            if x['id']>6:x['id']-=1
            if x['parent']>6:x['parent']-=1
        self.assertTrue(self.result(batch(ss))['structurally_valid'])
    def test_full_split_graph_may_match(self):
        r=fixture()
        for i in [4,5,6]:pick(r,i).update(graph='0x200',uid=9)
        self.result(r)
    def test_unknown_standalone_scope(self):
        s=span(1,0,'scheduler_sync','ggml_backend_sched_synchronize',1,2,
               ctx='0x0',lifetime=0,attempt=0,ubatch=0,tokens=-1,outputs=-1)
        r=self.result(batch([s],flush=(3,4)));self.assertEqual(r['unbound_context_spans'],1)
    def test_negative_decode_result_preserved(self):
        s=span(1,0,'decode','llama_context_decode',1,2,tokens=0,outputs=-1,result_known=1,result=-1)
        r=self.result(batch([s],flush=(3,4)));self.assertEqual(r['decode_results']['-1'],1)
    def test_decode_unwind_unknown_result(self):
        s=span(1,0,'decode','llama_context_decode',1,2,outputs=-1,unwind=1)
        self.result(batch([s],flush=(3,4)))
    def test_union_exclusive_no_double_count(self):
        r=self.result(fixture());self.assertEqual(r['per_process']['10']['covered_wall_union_us'],125)
        self.assertEqual(sum(x['exclusive_us'] for x in r['timeline']),125)
        self.assertGreater(sum(x['end_us']-x['start_us'] for x in r['timeline']),125)
    def test_interleaved_different_threads(self):
        a=fixture();b=copy.deepcopy(a)
        for x in b:
            x['pid']=11;x['tid']=21
        rows=[row for pair in zip(a,b) for row in pair]
        self.assertEqual(self.result(rows)['spans'],24)
    def test_missing_field(self):
        r=fixture();del pick(r,1)['parent'];self.fails(r)
    def test_extra_field(self):
        r=fixture();pick(r,1)['unused']=1;self.fails(r)
    def test_duplicate_field(self):
        rows=wire(fixture());rows[1]+=',pid=10'
        with self.assertRaises(ValueError):CHECK.analyze(rows)
    def test_leading_zero_integer(self):
        r=fixture();pick(r,1)['id']='01';self.fails(r)
    def test_negative_zero_integer(self):
        r=fixture();pick(r,1)['tokens']='-0';self.fails(r)
    def test_noncanonical_pointer(self):
        r=fixture();pick(r,1)['ctx']='0x0100';self.fails(r)
    def test_unknown_reason(self):
        r=fixture();pick(r,1)['reason']='pretend';self.fails(r)
    def test_wrong_direction(self):
        r=fixture();pick(r,8)['direction']='h2d';self.fails(r)
    def test_drop(self):
        r=fixture();r[0]['dropped']=1;self.fails(r)
    def test_exhausted(self):
        with self.assertRaises(ValueError):CHECK.analyze(['CUDA_HOST_PHASES,event=exhausted'])
    def test_allocation_failure_unconditional(self):
        with self.assertRaises(ValueError):CHECK.analyze(['CUDA_HOST_PHASES,v=1,event=allocation_failure'])
    def test_truncated_batch(self): self.fails(fixture()[:-1])
    def test_zero_records(self):
        with self.assertRaises(ValueError):CHECK.analyze(['unrelated'])
    def test_foreign_thread(self):
        r=fixture();pick(r,5)['thread']=2;self.fails(r)
    def test_foreign_tid(self):
        r=fixture();pick(r,5)['tid']=99;self.fails(r)
    def test_foreign_context(self):
        r=fixture();pick(r,5)['ctx']='0x999';self.fails(r)
    def test_foreign_attempt(self):
        r=fixture();pick(r,5)['attempt']=9;self.fails(r)
    def test_foreign_invocation(self):
        r=fixture();pick(r,5)['invocation']=99;self.fails(r)
    def test_foreign_split_uid(self):
        r=fixture();pick(r,5)['uid']=99;self.fails(r)
    def test_foreign_key(self):
        r=fixture();pick(r,6)['key']='0x999';self.fails(r)
    def test_unknown_early_child_rejected_for_bound_v1(self):
        r=fixture();pick(r,5).update(graph='0x200',uid=9,key='0x0',backend='0x0',device=-1,mode='unbound',invocation=0)
        self.fails(r)
    def test_crossing_siblings(self):
        r=fixture();pick(r,7)['start_us']=60;self.fails(r)
    def test_missing_parent(self):
        r=fixture();pick(r,5)['parent']=99;self.fails(r)
    def test_closed_parent_reopened(self):
        r=fixture();pick(r,7)['parent']=3;self.fails(r)
    def test_child_outside_parent(self):
        r=fixture();pick(r,6)['end_us']=61;self.fails(r)
    def test_end_before_start(self):
        r=fixture();pick(r,6)['end_us']=1;self.fails(r)
    def test_nonmonotonic_batch_clock(self):
        r=fixture();pick(r,9)['start_us']=105;self.fails(r)
    def test_flush_inside_root(self):
        r=fixture();r[0]['flush_start_us']=99;self.fails(r)
    def test_duplicate_span(self):
        r=fixture();r.insert(2,copy.deepcopy(pick(r,1)));self.fails(r)
    def test_missing_span(self):
        r=fixture();r.remove(pick(r,6));self.fails(r)
    def test_used_mismatch(self):
        r=fixture();next(x for x in r if x['event']=='batch' and x['batch']==2)['used']=11;self.fails(r)
    def test_budget_mismatch(self):
        r=fixture();r[0]['budget']=300000;self.fails(r)
    def test_aggregate_sum_mismatch(self):
        r=fixture();next(x for x in r if x['event']=='aggregate')['sum_us']+=1;self.fails(r)
    def test_aggregate_bytes_mismatch(self):
        r=fixture();next(x for x in r if x['event']=='aggregate')['bytes']+=1;self.fails(r)
    def test_duplicate_aggregate(self):
        r=fixture();i=next(i for i,x in enumerate(r) if x['event']=='aggregate');r.insert(i,copy.deepcopy(r[i]));self.fails(r)
    def test_missing_aggregate(self):
        r=fixture();r.remove(next(x for x in r if x['event']=='aggregate'));self.fails(r)
    def test_partial_start(self):
        r=fixture();r[0]['batch']=2;self.fails(r)
    def test_null_graph_with_nonzero_uid(self):
        r=fixture();pick(r,1)['uid']=1;self.fails(r)
    def test_foreign_getter_attempt(self):
        r=fixture()
        for i in [9,10,11,12]:pick(r,i)['attempt']=2
        self.fails(r)
    def test_invalid_result_known(self):
        r=fixture();pick(r,1)['result_known']=2;self.fails(r)
    def test_missing_decode_result(self):
        r=fixture();pick(r,1)['result_known']=0;self.fails(r)
    def test_same_process_interleaved_threads(self):
        a=fixture();b=copy.deepcopy(a)
        for x in b:
            x.update(tid=21,thread=4)
            if x['event']=='span':
                x.update(ctx='0x900',lifetime=5)
                if x['invocation']:x['invocation']=6
        self.assertEqual(self.result([r for pair in zip(a,b) for r in pair])['spans'],24)
    def test_legitimate_native_tid_reuse(self):
        r=fixture();s=span(1,0,'scheduler_sync','ggml_backend_sched_synchronize',200,210,
            thread=4,ctx='0x0',lifetime=0,attempt=0,tokens=-1,outputs=-1)
        self.result(r+batch([s],flush=(211,220)))
    def test_overlapping_native_tid_reuse(self):
        r=fixture();s=span(1,0,'scheduler_sync','ggml_backend_sched_synchronize',50,70,
            thread=4,ctx='0x0',lifetime=0,attempt=0,tokens=-1,outputs=-1)
        self.fails(r+batch([s],flush=(71,80)))
    def test_legitimate_context_pointer_reuse(self):
        r=fixture();s=span(13,0,'decode','llama_context_decode',200,210,lifetime=4,result_known=1,outputs=-1)
        self.result(r+batch([s],3,12,(211,220)))
    def test_overlapping_context_pointer_reuse(self):
        r=fixture();s=span(1,0,'decode','llama_context_decode',50,70,thread=4,tid=21,lifetime=5,result_known=1)
        self.fails(r+batch([s],flush=(71,80)))
    def test_zero_duration_nested_spans(self):
        r=fixture()
        for x in r:
            if x['event']=='span' and x['batch']==1:x.update(start_us=10,end_us=10)
        ss=[x for x in r if x['event']=='span' and x['batch']==1]
        self.assertEqual(self.result(batch(ss))['per_process']['10']['covered_wall_union_us'],0)
    def test_two_batch_headers_same_thread(self):
        r=fixture();r.insert(1,copy.deepcopy(r[0]));self.fails(r)
    def test_aggregate_before_spans(self):
        r=fixture();a=next(x for x in r if x['event']=='aggregate');r.remove(a);r.insert(1,a);self.fails(r)
    def test_footer_record_mismatch(self):
        r=fixture();next(x for x in r if x['event']=='end')['records']-=1;self.fails(r)
    def test_footer_clock_before_flush(self):
        r=fixture();next(x for x in r if x['event']=='end')['flush_end_us']=1;self.fails(r)
    def test_missing_replay_launch_recounted(self):
        r=fixture();ss=[x for x in r if x['event']=='span' and x['batch']==1 and x['id']!=6]
        for x in ss:
            if x['id']>6:x['id']-=1
            if x['parent']>6:x['parent']-=1
        self.fails(batch(ss))
    def test_wrong_nested_tokens(self):
        r=fixture();pick(r,5)['tokens']=512;self.fails(r)
    def test_graph_reused_uid_zero_new_invocation(self):
        r=fixture();ss=[copy.deepcopy(x) for x in r if x['event']=='span' and x['batch']==1]
        for x in ss:
            x['id']+=12
            if x['parent']:x['parent']+=12
            x['start_us']+=200;x['end_us']+=200;x['attempt']=2
            if x['invocation']:x['invocation']=4
        self.assertEqual(self.result(r+batch(ss,3,12,(301,310)))['graph_modes']['replay'],2)
    def test_input_upload_and_separate_buffer_wait(self):
        r=fixture();ss=[x for x in r if x['event']=='span' and x['batch']==1]
        for x in ss[2:]:
            x['id']+=5
            if x['parent']>=3:x['parent']+=5
        a=span(3,2,'input_setup','graph_set_inputs',21,29,inherit=ss[1])
        b=span(4,3,'input_one','set_input',22,28,inherit=a)
        c=span(5,4,'tensor_upload','ggml_backend_tensor_set',23,27,inherit=b,bytes=4)
        d=span(6,5,'cuda_buffer_copy','ggml_backend_cuda_buffer_set_tensor',24,25,inherit=c,
               bytes=4,direction='h2d',backend='0x600',device=0,stream='per_thread')
        e=span(7,5,'cuda_buffer_sync','ggml_backend_cuda_buffer_set_tensor',26,26,inherit=c,
               backend='0x600',device=0,stream='per_thread')
        result=self.result(batch(ss[:2]+[a,b,c,d,e]+ss[2:]))
        self.assertEqual(result['phases']['cuda_buffer_copy']['api_bytes'],4)
        self.assertEqual(result['phases']['cuda_buffer_sync']['count'],1)
    def test_unbound_standalone_cuda_compute(self):
        r=fixture();ss=[copy.deepcopy(pick(r,i)) for i in [4,5,6]]
        for i,s in enumerate(ss,1):
            s.update(id=i,parent=i-1,ctx='0x0',lifetime=0,attempt=0,ubatch=0,tokens=-1,outputs=-1)
        result=self.result(batch(ss,flush=(61,65)))
        self.assertEqual(result['unbound_context_spans'],3)
    def test_fake_backend_on_ordinary_scope(self):
        r=fixture();pick(r,2).update(backend='0x900',device=0);self.fails(r)
    def test_budget_exceeded(self):
        r=fixture();r[0]['used']=262145;self.fails(r)
    def test_negative_timestamp(self):
        r=fixture();pick(r,1)['start_us']=-1;self.fails(r)
    def test_multiple_markers(self):
        lines=wire(fixture());lines[1]+=' CUDA_HOST_PHASES,event=allocation_failure'
        with self.assertRaises(ValueError):CHECK.analyze(lines)
    def test_logits_child_byte_mismatch_recounted(self):
        ss=[r for r in fixture() if r['event']=='span' and r['batch']==1]
        ss[-1]['bytes']=4;self.fails(batch(ss))
    def test_logits_child_wrong_direction_recounted(self):
        ss=[r for r in fixture() if r['event']=='span' and r['batch']==1]
        ss[-1].update(reason='ggml_backend_cuda_set_tensor_async',direction='h2d');self.fails(batch(ss))
    def test_successful_decode_zero_tokens(self): self.fails(decode_fixture(tokens=0,outputs=0,parts=((1,0),)))
    def test_decode_outputs_exceed_tokens(self): self.fails(decode_fixture(tokens=1,outputs=2,parts=((1,1),)))
    def test_successful_multibatch(self):
        r=self.result(decode_fixture());self.assertEqual(r['decode_accounting'][0]['completed_ubatches'],2)
    def test_successful_multibatch_zero_outputs(self): self.result(decode_fixture(outputs=0,parts=((4,0),(2,0))))
    def test_successful_missing_tokens(self): self.fails(decode_fixture(parts=((4,2),)))
    def test_successful_excess_tokens_not_reservation_padding(self): self.fails(decode_fixture(parts=((4,1),(3,1))))
    def test_successful_output_sum_mismatch(self): self.fails(decode_fixture(parts=((4,0),(2,1))))
    def test_successful_no_ubatches_with_known_outputs(self): self.fails(decode_fixture(parts=()))
    def test_failed_partial_request(self):
        r=self.result(decode_fixture(result=-3,parts=((2,0),(2,1))))
        self.assertEqual(r['decode_accounting'][0]['completed_ubatches'],1)
        self.assertEqual(r['decode_accounting'][0]['last_ubatch_completion'],'not_established')
    def test_failed_before_ubatches(self): self.result(decode_fixture(result=-2,parts=()))
    def test_failed_early_unknown_outputs(self): self.result(decode_fixture(outputs=-1,result=-1,parts=()))
    def test_failed_excess_attempted_tokens(self): self.fails(decode_fixture(result=-3,parts=((4,1),(3,1))))
    def test_failed_excess_attempted_outputs(self): self.fails(decode_fixture(outputs=1,result=-3,parts=((4,1),(2,1))))
    def test_encode_fallback_unknown_outputs(self):
        r=self.result(decode_fixture(outputs=-1,parts=()))
        self.assertEqual(r['decode_accounting'][0]['kind'],'encode_fallback')
    def test_encode_fallback_must_not_have_decode_ubatches(self): self.fails(decode_fixture(outputs=-1))
    def test_zero_token_failure_must_be_minus_one(self): self.fails(decode_fixture(tokens=0,outputs=-1,result=-2,parts=()))
    def test_unwind_partial_ubatch_unknown_outputs(self):
        r=decode_fixture(unwind=1,parts=((2,0),(2,-1)))
        pick(r,3)['unwind']=1;self.result(r)
    def test_unwind_ubatch_cannot_precede_later_ubatch(self):
        r=decode_fixture(unwind=1);pick(r,2)['unwind']=1;self.fails(r)
    def test_success_cannot_unwind(self):
        r=decode_fixture();pick(r,2)['unwind']=1;self.fails(r)
    def test_logits_cpu_no_cuda_callback(self): self.result(request_fixture('logits_enqueue',[]))
    def test_logits_synchronous_fallback(self):
        self.result(request_fixture('logits_enqueue',[
            ('cuda_backend_sync','ggml_backend_cuda_synchronize','none',0),
            ('cuda_buffer_copy','ggml_backend_cuda_buffer_get_tensor','d2h',4),
            ('cuda_buffer_sync','ggml_backend_cuda_buffer_get_tensor','none',0)]))
    def test_logits_sync_then_cpu_fallback(self):
        self.result(request_fixture('logits_enqueue',[('cuda_backend_sync','ggml_backend_cuda_synchronize','none',0)]))
    def test_buffer_wait_without_copy_rejected(self):
        self.fails(request_fixture('logits_enqueue',[('cuda_buffer_sync','ggml_backend_cuda_buffer_get_tensor','none',0)]))
    def test_synchronous_fallback_wrong_bytes(self):
        self.fails(request_fixture('logits_enqueue',[
            ('cuda_buffer_copy','ggml_backend_cuda_buffer_get_tensor','d2h',8),
            ('cuda_buffer_sync','ggml_backend_cuda_buffer_get_tensor','none',0)]))
    def test_upload_cpu_zero_size(self): self.result(request_fixture('tensor_upload',[],size=0))
    def test_upload_zero_size_cannot_copy(self):
        self.fails(request_fixture('tensor_upload',[
            ('cuda_buffer_copy','ggml_backend_cuda_buffer_set_tensor','h2d',0),
            ('cuda_buffer_sync','ggml_backend_cuda_buffer_set_tensor','none',0)],size=0))
    def test_async_upload_zero_size_copy(self):
        self.result(request_fixture('tensor_upload_async',[
            ('cuda_async_copy','ggml_backend_cuda_set_tensor_async','h2d',0)],size=0))
    def test_logits_zero_size_generic_get_skips_buffer(self):
        self.fails(request_fixture('logits_enqueue',[
            ('cuda_buffer_copy','ggml_backend_cuda_buffer_get_tensor','d2h',0),
            ('cuda_buffer_sync','ggml_backend_cuda_buffer_get_tensor','none',0)],size=0))
    def test_upload_synchronous_wrong_bytes(self):
        self.fails(request_fixture('tensor_upload',[
            ('cuda_buffer_copy','ggml_backend_cuda_buffer_set_tensor','h2d',8),
            ('cuda_buffer_sync','ggml_backend_cuda_buffer_set_tensor','none',0)]))
    def test_async_upload_wrong_direction(self):
        self.fails(request_fixture('tensor_upload_async',[
            ('cuda_async_copy','ggml_backend_cuda_get_tensor_async','d2h',4)]))
    def test_async_upload_wrong_bytes(self):
        self.fails(request_fixture('tensor_upload_async',[
            ('cuda_async_copy','ggml_backend_cuda_set_tensor_async','h2d',8)]))
    def test_async_upload_fallback_nested_upload(self):
        root=span(1,0,'tensor_upload_async','ggml_backend_tensor_set_async',10,90,
                  ctx='0x0',lifetime=0,attempt=0,tokens=-1,outputs=-1,bytes=4)
        sync=span(2,1,'cuda_backend_sync','ggml_backend_cuda_synchronize',15,20,inherit=root,
                  backend='0x500',device=0,stream='backend_existing')
        upload=span(3,1,'tensor_upload','ggml_backend_tensor_set',25,80,inherit=root,bytes=4)
        cp=span(4,3,'cuda_buffer_copy','ggml_backend_cuda_buffer_set_tensor',30,40,inherit=upload,
                backend='0x500',device=0,stream='per_thread',direction='h2d',bytes=4)
        wait=span(5,3,'cuda_buffer_sync','ggml_backend_cuda_buffer_set_tensor',45,50,inherit=upload,
                  backend='0x500',device=0,stream='per_thread')
        self.result(batch([root,sync,upload,cp,wait]))
        upload['bytes']=8;self.fails(batch([root,sync,upload,cp,wait]))
    def test_transfer_unwind_allows_callback_prefix(self):
        self.result(request_fixture('tensor_upload',[
            ('cuda_buffer_copy','ggml_backend_cuda_buffer_set_tensor','h2d',4)],unwind=1))
    def test_transfer_cannot_mix_async_and_sync_paths(self):
        self.fails(request_fixture('logits_enqueue',[
            ('cuda_async_copy','ggml_backend_cuda_get_tensor_async','d2h',4),
            ('cuda_backend_sync','ggml_backend_cuda_synchronize','none',0)]))
    def test_transfer_cannot_use_2d_callback(self):
        self.fails(request_fixture('tensor_upload_async',[
            ('cuda_async_copy','ggml_backend_cuda_set_tensor_2d_async','h2d',4)]))
    def test_cli_success_failure_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'input.log';out=Path(tmp)/'out.json'
            log.write_text('\n'.join(wire(fixture()))+'\n',encoding='utf-8')
            args=[sys.executable,str(SCRIPT),'--log',str(log),'--output',str(out)]
            r=run_cli(args)
            self.assertEqual(r.returncode,0,r.stderr);data=out.read_bytes()
            self.assertFalse(json.loads(data)['external_coverage']['complete'])
            r=run_cli(args)
            self.assertEqual(r.returncode,2);self.assertEqual(out.read_bytes(),data)
            log.write_text('CUDA_HOST_PHASES,event=allocation_failure\n',encoding='utf-8')
            args[-1]=str(Path(tmp)/'failure.json');r=run_cli(args)
            self.assertEqual(r.returncode,2);self.assertFalse(json.loads(Path(args[-1]).read_text())['structurally_valid'])


if __name__=='__main__':
    program=unittest.main(verbosity=2,exit=False)
    print(json.dumps({'cli_runs':CLI_RUNS}))
    sys.exit(not program.result.wasSuccessful())
