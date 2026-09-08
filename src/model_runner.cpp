#include <algorithm>
#include <cassert>

#include <nlohmann/json.hpp>

#include "model_runner.h"
#include "cuda_context.h"
#include "model_config.h"
#include "forward_batch.h"
#include "workspace.h"
#include "batch_buffer.h"
#include "model/causal_lm.h"


namespace cllm
{

namespace
{

constexpr std::size_t INPUT_ALIGNMENT = 32;

}

ModelInput prepare_model_input(
    const std::vector<RequestData>& scheduled,
    int block_size
)
{
    assert(block_size > 0);

    ModelInput input;

    int num_tokens = 0;
    int max_num_blocks = 0;
    for (const RequestData& request : scheduled) {
        num_tokens += static_cast<int>(request.tokens_to_compute.size());
        max_num_blocks = std::max(
            max_num_blocks, static_cast<int>(request.allocated_blocks.size())
        );
    }

    const int num_reqs = static_cast<int>(scheduled.size());

    input.token_ids.reserve(num_tokens);
    input.positions.reserve(num_tokens);
    input.slot_mapping.reserve(num_tokens);
    input.query_start_loc.reserve(num_reqs + 1);
    input.seq_lens.reserve(num_reqs);
    input.block_table.assign(
        static_cast<std::size_t>(num_reqs) * max_num_blocks, 0
    );
    input.block_table_stride = max_num_blocks;

    input.query_start_loc.push_back(0);

    for (int i = 0; i < num_reqs; i++) {
        const RequestData& request = scheduled[i];
        const int num_query_tokens = static_cast<int>(request.tokens_to_compute.size());
        assert(num_query_tokens > 0);

        input.token_ids.insert(
            input.token_ids.end(),
            request.tokens_to_compute.begin(),
            request.tokens_to_compute.end()
        );

        for (int j = 0; j < num_query_tokens; j++) {
            const int position = request.position_offset + j;
            const int block = request.allocated_blocks[position / block_size];

            input.positions.push_back(position);
            input.slot_mapping.push_back(block * block_size + position % block_size);
        }

        input.query_start_loc.push_back(static_cast<int>(input.token_ids.size()));
        input.seq_lens.push_back(request.position_offset + num_query_tokens);

        std::copy(
            request.allocated_blocks.begin(),
            request.allocated_blocks.end(),
            input.block_table.begin() + static_cast<std::size_t>(i) * max_num_blocks
        );

        if (request.needs_sampling) {
            input.logits_indices.push_back(static_cast<int>(input.token_ids.size()) - 1);
            input.sampling_request_indices.push_back(i);
        }
    }

    return input;
}

// Everything below this line is device-side state. It lives here rather than in
// model_runner.h so Executor and EngineCore stay free of CUDA headers.
//
// The members are grouped by lifetime, not by type:
//   - ctx / weights   live for the whole process, created at construction
//   - kv_cache        created after profiling, once num_blocks is known
//   - the input_* and host_* buffers are per-step, but sized once from
//     max_num_scheduled_tokens and reused, so no step allocates.
//
// ctx is passed by reference to layers and kernel launchers; they never see
// Impl itself.
struct ModelRunner::Impl
{
    explicit Impl(const Config& config);

    void allocate_kv_cache(int num_blocks);

    std::vector<SamplerOutput> run_model(const std::vector<RequestData>& scheduled);

    Config config;

    ModelConfig model_config;

    CudaContext context;

    CausalLM model;

    // device buffer for activation
    Workspace workspace;

    // dataflow: vector in ModelInput -> PinnedBuffer -> CudaDeviceBuffer
    BatchBuffer batch_buffer;

    // allocated by allocate_kv_cache()
    KVCache kv_cache;
};

ModelRunner::Impl::Impl(const Config& config)
    : config(config),
      model_config(ModelConfig::load(config.model_path)),
      model(model_config, ModelWeights::load_from_safetensors(
          std::filesystem::path(config.model_path) / "model.safetensors",
          model_config
      )),
      workspace(Workspace::create(
          model_config,
          config.max_num_scheduled_tokens,
          config.max_num_seqs
      )),
      batch_buffer(BatchBuffer::create(config, model_config))
{
}

void ModelRunner::Impl::allocate_kv_cache(int num_blocks)
{
    kv_cache = KVCache::create(model_config, num_blocks, config.block_size);
}

std::vector<SamplerOutput> ModelRunner::Impl::run_model(const std::vector<RequestData>& scheduled)
{
    const ModelInput input = prepare_model_input(scheduled, config.block_size);

    const ForwardBatch batch = batch_buffer.upload(input, context);
    const auto aws = ActualWorkspace::create(workspace, batch.num_tokens, batch.num_sampling_reqs);
    model.forward(context, batch, kv_cache.view, aws);
    model.compute_logits(context, batch, aws);

    return {};
}

ModelRunner::ModelRunner(const Config& cfg)
    : impl_(std::make_unique<Impl>(cfg))
{
}

ModelRunner::~ModelRunner() noexcept = default;

std::size_t ModelRunner::profile_available_kv_cache_memory(float gpu_memory_utilization)
{
    // TODO: run a dummy forward pass at max_num_scheduled_tokens so peak
    // activation memory is touched, then derive the remaining budget from
    // cudaMemGetInfo.
    (void)gpu_memory_utilization;
    return 0;
}

std::size_t ModelRunner::kv_cache_block_bytes() const noexcept
{
    return KVCache::block_bytes(impl_->model_config, impl_->config.block_size);
}

void ModelRunner::allocate_kv_cache(int num_blocks)
{
    assert(num_blocks > 0);
    impl_->allocate_kv_cache(num_blocks);
}

std::vector<SamplerOutput> ModelRunner::run_model(const std::vector<RequestData>& scheduled)
{
    return impl_->run_model(scheduled);
}

}
