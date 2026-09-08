#include <algorithm>
#include <cassert>

#include "workspace.h"
#include "data_type.h"
#include "util.h"


namespace cllm
{

namespace
{

constexpr std::size_t ALIGNMENT = 256;

}

Workspace Workspace::create(const ModelConfig &config, int max_tokens, int max_seqs)
{
    assert(max_tokens > 0);
    assert(max_seqs > 0);

    const std::size_t hidden_size = config.hidden_size;
    const std::size_t q_size
        = static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
    const std::size_t kv_size = static_cast<std::size_t>(config.num_kv_heads) * config.head_dim;
    const std::size_t intermediate = config.intermediate_size;
    const std::size_t vocab_size = config.vocab_size;
    const std::size_t dtype_size = dtype_byte_size(config.dtype);

    auto reserve_tensor_bytes = [dtype_size](std::size_t& offset, std::size_t size) {
        offset = align_up<ALIGNMENT>(offset);
        offset += size * dtype_size;
    };

    std::size_t attn_workspace_size = 0;
    reserve_tensor_bytes(attn_workspace_size, max_tokens * hidden_size); // hidden
    reserve_tensor_bytes(attn_workspace_size, max_tokens * hidden_size); // residual
    reserve_tensor_bytes(attn_workspace_size, max_tokens * (q_size + 2 * kv_size)); // qkv
    reserve_tensor_bytes(attn_workspace_size, max_tokens * q_size); // attn_out

    std::size_t ffn_workspace_size = 0;
    reserve_tensor_bytes(ffn_workspace_size, max_tokens * hidden_size); // hidden
    reserve_tensor_bytes(ffn_workspace_size, max_tokens * hidden_size); // residual
    reserve_tensor_bytes(ffn_workspace_size, max_tokens * 2 * intermediate); // gate_up

    std::size_t logits_workspace_size = 0;
    reserve_tensor_bytes(logits_workspace_size, max_seqs * hidden_size); // sampling_hidden
    reserve_tensor_bytes(logits_workspace_size, max_seqs * vocab_size); // logits

    std::size_t required_byte_size = std::max({
        attn_workspace_size,
        ffn_workspace_size,
        logits_workspace_size
    });

    CudaDeviceBuffer data(required_byte_size);
    auto* base = data.data<std::byte>();

    auto place_view = [&](std::size_t& offset, std::size_t rows, std::size_t cols) {
        offset = align_up<ALIGNMENT>(offset);
        auto* device_ptr = base + offset;
        offset += rows * cols * dtype_size;
        return make_tensor<2>(
            device_ptr,
            config.dtype,
            {static_cast<int>(rows), static_cast<int>(cols)}
        );
    };

    attn_workspace_size = 0;
    logits_workspace_size = 0;
    
    Workspace workspace {};
    workspace.hidden = place_view(attn_workspace_size, max_tokens, hidden_size);
    workspace.residual = place_view(attn_workspace_size, max_tokens, hidden_size);

    ffn_workspace_size = attn_workspace_size;
    // attn
    workspace.qkv = place_view(attn_workspace_size, max_tokens, q_size + 2 * kv_size);
    workspace.attn_out = place_view(attn_workspace_size, max_tokens, q_size);
    // ffn
    workspace.gate_up = place_view(ffn_workspace_size, max_tokens, 2 * intermediate);
    // compute logits
    workspace.sampling_hidden = place_view(logits_workspace_size, max_seqs, hidden_size);
    workspace.logits = place_view(logits_workspace_size, max_seqs, vocab_size);

    workspace.data = std::move(data);

    return workspace;
}

ActualWorkspace ActualWorkspace::create(
    const Workspace &workspace,
    int num_tokens,
    int num_sampling_seqs
)
{
    return ActualWorkspace{
        .hidden = make_tensor<2>(
            workspace.hidden.device_ptr,
            workspace.hidden.dtype,
            {num_tokens, workspace.hidden.shape[1]}
        ),
        .residual = make_tensor<2>(
            workspace.residual.device_ptr,
            workspace.residual.dtype,
            {num_tokens, workspace.residual.shape[1]}
        ),
        .qkv = make_tensor<2>(
            workspace.qkv.device_ptr,
            workspace.qkv.dtype,
            {num_tokens, workspace.qkv.shape[1]}
        ),
        .attn_out = make_tensor<2>(
            workspace.attn_out.device_ptr,
            workspace.attn_out.dtype,
            {num_tokens, workspace.attn_out.shape[1]}
        ),
        .gate_up = make_tensor<2>(
            workspace.gate_up.device_ptr,
            workspace.gate_up.dtype,
            {num_tokens, workspace.gate_up.shape[1]}
        ),
        .gated = make_tensor<2>(
            workspace.gate_up.device_ptr,
            workspace.gate_up.dtype,
            {num_tokens, workspace.gate_up.shape[1] / 2},
            {workspace.gate_up.shape[1], 1}
        ),
        .sampling_hidden = (num_sampling_seqs != 0)
                           ? make_tensor<2>(
                                 workspace.sampling_hidden.device_ptr,
                                 workspace.sampling_hidden.dtype,
                                 {num_sampling_seqs, workspace.sampling_hidden.shape[1]}
                             )
                           : Tensor<2>{},
        .logits = (num_sampling_seqs != 0)
                  ? make_tensor<2>(
                        workspace.logits.device_ptr,
                        workspace.logits.dtype,
                        {num_sampling_seqs, workspace.logits.shape[1]}
                    )
                  : Tensor<2>{}
    };
}

}