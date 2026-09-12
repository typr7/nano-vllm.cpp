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

    std::size_t profile_available_kv_cache_memory(float gpu_memory_utilization);
    void allocate_kv_cache(int num_blocks);

    std::size_t kv_cache_block_bytes() const noexcept;
    ModelConfig model_config() const noexcept;

    void execute(const std::vector<ScheduledRequest>& scheduled);
    std::vector<SampledToken> finish();

private:
    ModelRunner model_runner_;
};

}
