#pragma once

// Internal host-only tracing bridge. One implementation in ggml-base owns the
// clock, identities and TLS across llama and backend DLLs. Not a public op/API.
#include "ggml.h"
#include <cstdint>

GGML_API bool ggml_host_trace_enabled();
GGML_API uint64_t ggml_host_trace_identity();
GGML_API void * ggml_host_trace_begin(const char * phase, const char * reason);
GGML_API void ggml_host_trace_end(void * record);
GGML_API void ggml_host_trace_context(void *, const void *, uint64_t, uint64_t, uint64_t, int64_t, int64_t);
GGML_API void ggml_host_trace_graph(void *, const void *, uint64_t, const void *, const void *, int, const char *, bool);
GGML_API void ggml_host_trace_backend(void *, const void *, int, const char *);
GGML_API void ggml_host_trace_transfer(void *, uint64_t, const char *);
GGML_API void ggml_host_trace_outputs(void *, int64_t);
GGML_API void ggml_host_trace_result(void *, int64_t);

struct ggml_host_trace_scope {
    void * record;
    explicit ggml_host_trace_scope(const char * phase, const char * reason = "none") :
        record(ggml_host_trace_enabled() ? ggml_host_trace_begin(phase, reason) : nullptr) {}
    ~ggml_host_trace_scope() { if (record) { ggml_host_trace_end(record); } }
    ggml_host_trace_scope(const ggml_host_trace_scope &) = delete;
    ggml_host_trace_scope & operator=(const ggml_host_trace_scope &) = delete;
    explicit operator bool() const { return record != nullptr; }
    template<class T> T result(T value) {
        if (record) { ggml_host_trace_result(record, static_cast<int64_t>(value)); }
        return value;
    }
};
