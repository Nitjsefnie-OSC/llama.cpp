#pragma once

#include "common.cuh"

#include <cstdlib>
#include <vector>

static bool ggml_cuda_timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("DEBUG_CUDA_TIMING");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

struct ggml_cuda_timing {
    struct entry {
        const ggml_tensor * node;
        cudaEvent_t begin;
        cudaEvent_t end;
    };

    bool enabled = ggml_cuda_timing_enabled();
    cudaStream_t stream;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    std::vector<entry> entries;

    explicit ggml_cuda_timing(cudaStream_t stream) : stream(stream) {
        if (enabled) {
            CUDA_CHECK(cudaEventCreate(&begin));
            CUDA_CHECK(cudaEventCreate(&end));
            CUDA_CHECK(cudaEventRecord(begin, stream));
        }
    }

    struct scope {
        ggml_cuda_timing & timing;
        cudaStream_t stream;
        size_t index = 0;

        scope(ggml_cuda_timing & timing, const ggml_tensor * node, cudaStream_t stream)
            : timing(timing), stream(stream) {
            if (timing.enabled) {
                entry item{node, nullptr, nullptr};
                CUDA_CHECK(cudaEventCreate(&item.begin));
                CUDA_CHECK(cudaEventCreate(&item.end));
                index = timing.entries.size();
                timing.entries.push_back(item);
                CUDA_CHECK(cudaEventRecord(item.begin, stream));
            }
        }

        ~scope() {
            if (timing.enabled) {
                CUDA_CHECK(cudaEventRecord(timing.entries[index].end, stream));
            }
        }
    };

    ~ggml_cuda_timing() {
        if (!enabled) {
            return;
        }
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float graph_ms = 0.0f;
        float sum_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&graph_ms, begin, end));
        for (const entry & item : entries) {
            CUDA_CHECK(cudaEventSynchronize(item.end));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, item.begin, item.end));
            sum_ms += ms;
            const ggml_tensor * node = item.node;
            const ggml_tensor * src = node->src[0];
            GGML_LOG_INFO("CUDA_TIMING,%s,%s,%s,%lld,%lld,%lld,%.6f\n",
                          ggml_op_name(node->op), node->name, src ? ggml_type_name(src->type) : "none",
                          (long long) (src ? src->ne[0] : 0), (long long) node->ne[0],
                          (long long) node->ne[1], ms);
            CUDA_CHECK(cudaEventDestroy(item.begin));
            CUDA_CHECK(cudaEventDestroy(item.end));
        }
        // With concurrent streams the operation sum may exceed graph elapsed time.
        GGML_LOG_INFO("CUDA_TIMING_TOTAL,%zu,%.6f,%.6f\n", entries.size(), graph_ms, sum_ms);
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
    }
};
