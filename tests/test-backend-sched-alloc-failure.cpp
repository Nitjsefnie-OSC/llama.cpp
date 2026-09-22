#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>

// Two mock CPU backends exercise the scheduler's multi-buffer allocator path
// without loading devices or computing a graph. Only buffer allocation can fail.
struct fixture {
    bool fail = false;
    int failures = 0;
    int synchronizations = 0;
    ggml_backend_device devices[2] = {};
    ggml_backend_buffer_type bufts[2] = {};
    ggml_backend backends[2] = {};
    ggml_backend_sched_t sched;

    fixture() {
        ggml_backend_t backend_ptrs[2];
        ggml_backend_buffer_type_t buft_ptrs[2];
        for (int i = 0; i < 2; ++i) {
            auto & dev = devices[i];
            dev.iface.get_name = [](ggml_backend_dev_t) { return "mock-cpu"; };
            dev.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
            dev.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
            dev.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t) { return true; };

            auto & buft = bufts[i];
            buft.device = &dev;
            buft.context = this;
            buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "mock-failing-buffer"; };
            buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 32; };
            buft.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
            buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
                auto * self = static_cast<fixture *>(type->context);
                if (self->fail) {
                    ++self->failures;
                    return (ggml_backend_buffer_t) nullptr;
                }
                auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
                if (buffer) {
                    buffer->buft = type;
                }
                return buffer;
            };

            auto & backend = backends[i];
            backend.device = &dev;
            backend.context = this;
            backend.iface.get_name = [](ggml_backend_t) { return "mock-cpu"; };
            backend.iface.synchronize = [](ggml_backend_t value) {
                ++static_cast<fixture *>(value->context)->synchronizations;
            };
            backend_ptrs[i] = &backend;
            buft_ptrs[i] = &buft;
        }
        sched = ggml_backend_sched_new(backend_ptrs, buft_ptrs, 2, 32, false, false);
    }

    ~fixture() {
        ggml_backend_sched_free(sched);
    }
};

struct graph {
    ggml_context * ctx;
    ggml_cgraph * value;

    explicit graph(int64_t width) {
        const ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false),
                                         nullptr, true };
        ctx = ggml_init(params);
        GGML_ASSERT(ctx);
        value = ggml_new_graph_custom(ctx, 32, false);
        auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_set_input(input);
        ggml_build_forward_expand(value, ggml_dup(ctx, input));
    }

    ~graph() {
        ggml_free(ctx);
    }
};

static bool run_case(bool growth, bool fail) {
    graph small(16);
    graph large(65536);
    fixture f;
    if (growth && !ggml_backend_sched_alloc_graph(f.sched, small.value)) {
        std::fprintf(stderr, "initial allocation unexpectedly failed\n");
        return false;
    }
    ggml_backend_sched_reset(f.sched);
    f.fail = fail;
    const int sync_before = f.synchronizations;
    const bool allocated = ggml_backend_sched_alloc_graph(f.sched, large.value);
    if (allocated == fail || (fail && f.failures != 1) || f.synchronizations < sync_before + 2) {
        std::fprintf(stderr, "unexpected result: allocated=%d failures=%d synchronizations=%d\n",
                     allocated, f.failures, f.synchronizations - sync_before);
        return false;
    }
    std::printf("PASS %s %s\n", growth ? "growth" : "initial", fail ? "failure" : "success");
    if (fail) {
        f.fail = false;
        ggml_backend_sched_reset(f.sched);
        if (!ggml_backend_sched_alloc_graph(f.sched, large.value)) {
            std::fprintf(stderr, "retry after allocation failure did not recover\n");
            return false;
        }
        std::printf("PASS %s retry\n", growth ? "growth" : "initial");
    }
    return true;
}

int main(int argc, char ** argv) {
    // Individual cases let the regression demonstrate the unpatched crash in a
    // child process without hiding whether initial allocation or growth failed.
    if (argc == 2) {
        if (std::strcmp(argv[1], "initial-failure") == 0) { return run_case(false, true) ? 0 : 1; }
        if (std::strcmp(argv[1], "growth-failure") == 0)  { return run_case(true, true) ? 0 : 1; }
        return 2;
    }
    return run_case(false, false) && run_case(true, false) &&
           run_case(false, true) && run_case(true, true) ? 0 : 1;
}
