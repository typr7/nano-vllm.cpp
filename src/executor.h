#pragma once

#include <cstddef>
#include <vector>

#include "config.h"
#include "model_runner.h"
#include "request.h"


namespace cllm
{

class Executor
{
public:
    explicit Executor(const Config& cfg);
    ~Executor() noexcept;

    // Forwarded to ModelRunner, which is the only place that knows the KV cache
    // layout. Call in this order, once, before the first execute().
    std::size_t available_memory_for_kv_cache(float gpu_memory_utilization);
    std::size_t kv_cache_block_bytes() const noexcept;
    void allocate_kv_cache(int num_blocks);

    std::vector<SamplerOutput> execute(const std::vector<RequestData>& scheduled);

private:
    ModelRunner model_runner_;
};

}
