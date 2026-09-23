#include "ggml-host-trace.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <inttypes.h>
#include <new>
#include <memory>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <pthread.h>
#endif

namespace {
constexpr uint64_t capacity = 4096;
constexpr uint64_t budget = 262144;
struct binding {
    const void * ctx = nullptr;
    uint64_t lifetime = 0, attempt = 0, ubatch = 0;
    int64_t tokens = -1, outputs = -1;
    const void * graph = nullptr;
    uint64_t uid = 0;
    const void * key = nullptr;
    const void * backend = nullptr;
    int device = -1;
    const char * mode = "unbound";
    uint64_t invocation = 0;
};
struct record {
    binding b;
    record * parent = nullptr;
    uint64_t id = 0, bytes = 0;
    const char * phase = nullptr;
    const char * reason = nullptr;
    const char * direction = "none";
    const char * stream = "unknown";
    int64_t start = 0, end = 0, result = 0;
    int exceptions = 0;
    bool result_known = false, unwind = false;
};
struct state {
    uint64_t pid = 0, tid = 0, thread = 0, batch = 0, next_id = 0;
    uint64_t used = 0, count = 0, dropped = 0;
    bool exhausted = false;
    record * current = nullptr;
    record records[capacity];
};
// A null TLS pointer is the entire disabled state. Allocation is enabled-only.
thread_local std::unique_ptr<state> tls;
std::atomic<uint64_t> identities { 0 };

void native_identity(uint64_t & pid, uint64_t & tid) {
#if defined(_WIN32)
    pid = GetCurrentProcessId(); tid = GetCurrentThreadId();
#elif defined(__linux__)
    pid = getpid(); tid = syscall(SYS_gettid);
#elif defined(__APPLE__)
    pid = getpid(); pthread_threadid_np(nullptr, &tid);
#else
    // Unsupported native TID is explicitly unbound, never a hashed C++ ID.
    pid = 0; tid = 0;
#endif
}

void flush(state & s) {
    // Called only after the root end timestamp, with no measured span active.
    const int64_t begin = ggml_time_us();
    const int valid = s.dropped == 0 && s.pid != 0 && s.tid != 0;
    GGML_LOG_INFO("CUDA_HOST_PHASES,v=1,event=batch,pid=%" PRIu64 ",tid=%" PRIu64 ",thread=%" PRIu64
                  ",batch=%" PRIu64 ",clock=ggml_time_us,unit=us,records=%" PRIu64 ",dropped=%" PRIu64
                  ",budget=%" PRIu64 ",used=%" PRIu64 ",valid=%d,flush_start_us=%" PRId64 "\n",
                  s.pid, s.tid, s.thread, s.batch, s.count, s.dropped, budget, s.used, valid, begin);
    for (uint64_t i = 0; i < s.count; ++i) {
        const record & r = s.records[i];
        const binding & b = r.b;
        GGML_LOG_INFO("CUDA_HOST_PHASES,v=1,event=span,pid=%" PRIu64 ",tid=%" PRIu64 ",thread=%" PRIu64
                      ",batch=%" PRIu64 ",id=%" PRIu64 ",parent=%" PRIu64 ",phase=%s,reason=%s,start_us=%" PRId64
                      ",end_us=%" PRId64 ",ctx=0x%" PRIxPTR ",lifetime=%" PRIu64 ",attempt=%" PRIu64
                      ",ubatch=%" PRIu64 ",tokens=%" PRId64 ",outputs=%" PRId64 ",graph=0x%" PRIxPTR
                      ",uid=%" PRIu64 ",key=0x%" PRIxPTR ",backend=0x%" PRIxPTR ",device=%d,mode=%s,invocation=%" PRIu64
                      ",bytes=%" PRIu64 ",direction=%s,stream=%s,result_known=%d,result=%" PRId64 ",unwind=%d\n",
                      s.pid, s.tid, s.thread, s.batch, r.id, r.parent ? r.parent->id : 0, r.phase, r.reason,
                      r.start, r.end, (uintptr_t) b.ctx, b.lifetime, b.attempt, b.ubatch, b.tokens, b.outputs,
                      (uintptr_t) b.graph, b.uid, (uintptr_t) b.key, (uintptr_t) b.backend, b.device, b.mode,
                      b.invocation, r.bytes, r.direction, r.stream, (int) r.result_known, r.result, (int) r.unwind);
    }
    uint64_t aggregates = 0;
    // Bounded scan avoids another allocation/table or unbounded label storage.
    for (uint64_t i = 0; i < s.count; ++i) {
        const record & r = s.records[i];
        auto same = [&r](const record & q) {
            return !strcmp(r.phase, q.phase) && !strcmp(r.reason, q.reason) && !strcmp(r.direction, q.direction);
        };
        bool seen = false;
        for (uint64_t j = 0; j < i; ++j) { if (same(s.records[j])) { seen = true; break; } }
        if (seen) { continue; }
        uint64_t count = 0, bytes = 0;
        int64_t sum = 0, maximum = 0;
        for (uint64_t j = i; j < s.count; ++j) {
            if (!same(s.records[j])) { continue; }
            const record & q = s.records[j];
            ++count; bytes += q.bytes;
            const int64_t duration = q.end - q.start;
            sum += duration; if (duration > maximum) { maximum = duration; }
        }
        ++aggregates;
        GGML_LOG_INFO("CUDA_HOST_PHASES,v=1,event=aggregate,pid=%" PRIu64 ",tid=%" PRIu64 ",thread=%" PRIu64
                      ",batch=%" PRIu64 ",phase=%s,reason=%s,direction=%s,count=%" PRIu64
                      ",bytes=%" PRIu64 ",sum_us=%" PRId64 ",max_us=%" PRId64 "\n",
                      s.pid, s.tid, s.thread, s.batch, r.phase, r.reason, r.direction, count, bytes, sum, maximum);
    }
    GGML_LOG_INFO("CUDA_HOST_PHASES,v=1,event=end,pid=%" PRIu64 ",tid=%" PRIu64 ",thread=%" PRIu64
                  ",batch=%" PRIu64 ",records=%" PRIu64 ",aggregates=%" PRIu64 ",dropped=%" PRIu64
                  ",valid=%d,flush_end_us=%" PRId64 "\n",
                  s.pid, s.tid, s.thread, s.batch, s.count, aggregates, s.dropped, valid, ggml_time_us());
    s.count = 0; s.dropped = 0;
}
} // namespace

bool ggml_host_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("DEBUG_CUDA_HOST_PHASES");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

uint64_t ggml_host_trace_identity() {
    return identities.fetch_add(1, std::memory_order_relaxed) + 1;
}

void * ggml_host_trace_begin(const char * phase, const char * reason) {
    if (!tls) {
        tls.reset(new (std::nothrow) state);
        if (!tls) {
            GGML_LOG_ERROR("CUDA_HOST_PHASES,v=1,event=allocation_failure\n");
            return nullptr;
        }
        native_identity(tls->pid, tls->tid);
        tls->thread = ggml_host_trace_identity();
    }
    state & s = *tls;
    if (s.exhausted) { return nullptr; }
    if (!s.current && s.used >= budget) {
        s.exhausted = true;
        GGML_LOG_INFO("CUDA_HOST_PHASES,v=1,event=exhausted,pid=%" PRIu64 ",tid=%" PRIu64 ",thread=%" PRIu64
                      ",batch=%" PRIu64 ",budget=%" PRIu64 ",used=%" PRIu64 ",dropped_at_least=1,truncated=1\n",
                      s.pid, s.tid, s.thread, s.batch, budget, s.used);
        return nullptr;
    }
    if (s.count == capacity || s.used == budget) { ++s.dropped; return nullptr; }
    if (!s.current) { ++s.batch; }
    record & r = s.records[s.count++];
    r = record();
    r.parent = s.current;
    if (s.current) { r.b = s.current->b; }
    r.id = ++s.next_id; ++s.used;
    r.phase = phase; r.reason = reason;
    r.exceptions = std::uncaught_exceptions();
    r.start = ggml_time_us();
    s.current = &r;
    return &r;
}

void ggml_host_trace_end(void * opaque) {
    record & r = *static_cast<record *>(opaque);
    r.end = ggml_time_us();
    r.unwind = std::uncaught_exceptions() > r.exceptions;
    tls->current = r.parent;
    if (!tls->current) { flush(*tls); }
}

void ggml_host_trace_context(void * p, const void * ctx, uint64_t lifetime, uint64_t attempt,
                             uint64_t ubatch, int64_t tokens, int64_t outputs) {
    record & r = *static_cast<record *>(p);
    binding & b = r.b;
    // Readiness called inside a ubatch keeps that parent's token/ubatch identity.
    // A root getter binds to the last decode attempt with unknown token count.
    if (r.parent && b.ctx == ctx && !strcmp(r.phase, "context_sync")) { return; }
    b.ctx = ctx; b.lifetime = lifetime; b.attempt = attempt; b.ubatch = ubatch; b.tokens = tokens; b.outputs = outputs;
}
void ggml_host_trace_graph(void * p, const void * graph, uint64_t uid, const void * key,
                          const void * backend, int device, const char * mode, bool new_invocation) {
    binding & b = static_cast<record *>(p)->b;
    b.graph = graph; b.uid = uid; b.key = key; b.backend = backend; b.device = device; b.mode = mode;
    if (new_invocation) { b.invocation = ggml_host_trace_identity(); }
}
void ggml_host_trace_backend(void * p, const void * backend, int device, const char * stream) {
    record & r = *static_cast<record *>(p);
    r.b.backend = backend; r.b.device = device; r.stream = stream;
}
void ggml_host_trace_transfer(void * p, uint64_t bytes, const char * direction) {
    record & r = *static_cast<record *>(p); r.bytes = bytes; r.direction = direction;
}
void ggml_host_trace_outputs(void * p, int64_t outputs) { static_cast<record *>(p)->b.outputs = outputs; }
void ggml_host_trace_result(void * p, int64_t result) {
    record & r = *static_cast<record *>(p); r.result = result; r.result_known = true;
}
