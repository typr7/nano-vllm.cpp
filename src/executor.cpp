#include "executor.h"


namespace cllm
{

Executor::Executor(const Config& cfg)
    : model_runner_(cfg)
{
}

Executor::~Executor() noexcept = default;

std::size_t Executor::profile_available_kv_cache_memory(float gpu_memory_utilization)
{
    return model_runner_.profile_available_kv_cache_memory(gpu_memory_utilization);
}

void Executor::allocate_kv_cache(int num_blocks)
{
    model_runner_.allocate_kv_cache(num_blocks);
}

std::size_t Executor::kv_cache_block_bytes() const noexcept
{
    return model_runner_.kv_cache_block_bytes();
}

ModelConfig Executor::model_config() const noexcept
{
    return model_runner_.model_config();
}

void Executor::execute(const std::vector<ScheduledRequest>& scheduled)
{
    return model_runner_.run_model(scheduled);
}

std::vector<SampledToken> Executor::finish()
{
    return model_runner_.finish();
}

}
