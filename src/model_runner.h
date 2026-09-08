#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "request.h"


namespace cllm
{

// One token sampled for one request during one step.
//
// A request in the middle of a chunked prefill produces nothing, so this is
// generally shorter than the scheduled batch.
struct SamplerOutput
{
    std::string request_id;
    int token_id;
};

// Host-side, flattened description of one scheduled step.
//
// Prefill and decode are not separated: a decode request is just a request with
// a single query token. That is what a varlen attention kernel wants, and it
// keeps a mixed chunked-prefill batch down to one launch per operator.
//
// Given scheduled = [A: 3 tokens from position 0, holding block 7,
//                    B: 1 token from position 5, holding block 2]
// with block_size 16:
//
//   token_ids       = [a0, a1, a2, b5]      // num_tokens == 4
//   positions       = [ 0,  1,  2,  5]
//   query_start_loc = [ 0,  3,  4]          // num_reqs + 1, exclusive scan
//   seq_lens        = [ 3,  6]              // context length AFTER this step
//   slot_mapping    = [112, 113, 114,  37]  // block * block_size + offset
//   block_table     = [7, 2]                // row-major, stride 1 here
//   logits_indices  = [2, 3]                // last query token of each sampler
//
// slot_mapping is deliberately layer-independent: the same slot index addresses
// every layer's K and V once the layer stride is applied.
struct ModelInput
{
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> slot_mapping;
    std::vector<int> query_start_loc;
    std::vector<int> seq_lens;
    // num_reqs rows of block_table_stride entries, padded with zeros.
    std::vector<int> block_table;
    // Positions within token_ids whose logits must be materialized, and the
    // index into the scheduled batch that each one belongs to. Parallel arrays.
    std::vector<int> logits_indices;
    // this will not be uploaded to device buffer
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

// Flattens a scheduled batch into the layout above. Pure host-side logic, kept
// out of ModelRunner so it can be tested without a GPU.
ModelInput prepare_model_input(
    const std::vector<RequestData>& scheduled,
    int block_size
);

// Owns the model weights, the paged KV cache, and every CUDA resource needed to
// run a forward pass. Everything above it (Executor, Scheduler, EngineCore) is
// plain host C++; the pimpl keeps CUDA headers from leaking upward.
class ModelRunner
{
public:
    // Parses the Hugging Face config, uploads the SafeTensors weights, and
    // creates the stream and cuBLAS handle. The KV cache is NOT allocated here:
    // its size is only known after profiling.
    explicit ModelRunner(const Config& cfg);
    ~ModelRunner() noexcept;

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // --- one-time initialization, in this order ---

    // Runs a dummy forward pass at max_num_scheduled_tokens so peak activation
    // memory is actually touched, then reports how many bytes are left for the
    // KV cache under the given utilization target.
    std::size_t profile_available_kv_cache_memory(float gpu_memory_utilization);

    // Bytes one block occupies across all layers, both K and V:
    //   2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype
    // The layer count lives here so callers cannot forget it.
    std::size_t kv_cache_block_bytes() const noexcept;

    // Allocates the paged KV cache as one contiguous region shaped
    //   [num_hidden_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
    // and hands each layer its K/V base pointers.
    void allocate_kv_cache(int num_blocks);

    // --- steady state ---

    // Runs one forward pass over the scheduled batch and samples one token for
    // every request whose prefill completed. Blocks until the sampled tokens
    // are back on the host.
    std::vector<SamplerOutput> run_model(const std::vector<RequestData>& scheduled);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
