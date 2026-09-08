#include <algorithm>

#include "batch_buffer.h"
#include "util.h"


namespace cllm
{

namespace
{

constexpr std::size_t ALIGNMENT = 32;

}

BatchBuffer BatchBuffer::create(const Config &config, const ModelConfig &model_config)
{
    // construct a max input to determine the buffer size
    const int max_tokens = config.max_num_scheduled_tokens;
    const int max_seqs = config.max_num_seqs;
    const int block_size = config.block_size;
    const int max_model_len = model_config.max_model_len;
    const int max_block_table_stride = (max_model_len + block_size - 1) / block_size;

    std::size_t buffer_size = 0;
    auto reserve = [&buffer_size](std::size_t size) {
        buffer_size = align_up<ALIGNMENT>(buffer_size);
        buffer_size += size * sizeof(int);
    };
    reserve(max_tokens); // token_ids
    reserve(max_tokens); // positions
    reserve(max_tokens); // slot_mapping
    reserve(max_seqs + 1); // query_start_loc
    reserve(max_seqs); // seq_lens
    reserve(static_cast<std::size_t>(max_seqs) * max_block_table_stride); // block_table
    reserve(max_seqs); // logits_indices

    BatchBuffer buffer;
    buffer.device_.resize(buffer_size);
    buffer.pinned_.resize(buffer_size);
    return buffer;
}

ForwardBatch BatchBuffer::upload(const ModelInput& input, const CudaContext& context)
{
    std::size_t upload_size = 0;
    auto upload_to_pinned = [&](const std::vector<int>& in) -> const int* {
        if (in.empty()) {
            return nullptr;
        }

        upload_size = align_up<ALIGNMENT>(upload_size);

        std::size_t byte_size = in.size() * sizeof(int);
        pinned_.upload_at(upload_size, in.data(), byte_size);

        const int* device_ptr = reinterpret_cast<int*>(
            device_.data<std::byte>() + upload_size
        );
        upload_size += byte_size;
        return device_ptr;
    };

    ForwardBatch batch{
        .token_ids = upload_to_pinned(input.token_ids),
        .positions = upload_to_pinned(input.positions),
        .slot_mapping = upload_to_pinned(input.slot_mapping),
        .query_start_loc = upload_to_pinned(input.query_start_loc),
        .seq_lens = upload_to_pinned(input.seq_lens),
        .block_table = upload_to_pinned(input.block_table),
        .logits_indices = upload_to_pinned(input.logits_indices),

        .block_table_stride = input.block_table_stride,
        .num_tokens = input.num_tokens(),
        .num_reqs = input.num_reqs(),
        .num_sampling_reqs = static_cast<int>(input.logits_indices.size()),
        .max_query_len = 0,
        .max_seq_len = *std::max_element(input.seq_lens.begin(), input.seq_lens.end())
    };

    for (int i = 0; i < input.num_reqs(); i++) {
        batch.max_query_len = std::max(
            batch.max_query_len,
            input.query_start_loc[i + 1] - input.query_start_loc[i]
        );
    }

    device_.upload_async(pinned_.data(), upload_size, context.stream());

    return batch;
}

}