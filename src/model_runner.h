#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "config.h"
#include "request.h"
#include "model_config.h"


namespace cllm
{

struct ModelInput
{
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> slot_mapping;
    std::vector<int> query_start_loc;
    std::vector<int> seq_lens;
    std::vector<int> block_table;
    std::vector<int> logits_indices;
    std::vector<SampleParams> sample_params;

    // will not be passed to device
    std::vector<int> sampling_request_indices;

    int block_table_stride = 0;

    int num_tokens() const noexcept
    {
        return static_cast<int>(token_ids.size());
    }

    int num_reqs() const noexcept
    {
        return static_cast<int>(seq_lens.size());
    }
};

class ModelRunner
{
public:
    explicit ModelRunner(const Config& cfg);
    ~ModelRunner() noexcept;

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    std::size_t profile_available_kv_cache_memory(float gpu_memory_utilization);
    void allocate_kv_cache(int num_blocks);

    std::size_t kv_cache_block_bytes() const noexcept;
    ModelConfig model_config() const noexcept;

    void run_model(const std::vector<ScheduledRequest>& scheduled);
    std::vector<SampledToken> finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Flattens a scheduled batch into the layout above. Pure host-side logic, kept
// out of ModelRunner so it can be tested without a GPU.
ModelInput prepare_model_input(
    const std::vector<ScheduledRequest>& scheduled,
    int block_size
);

}
