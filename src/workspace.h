#pragma once

#include "tensor.h"
#include "cuda_device_buffer.h"
#include "model_config.h"


namespace cllm
{

struct Workspace
{
    Tensor<2> hidden; // [T, H] 1, 4, 7
    Tensor<2> residual; // [T, H] 0, 5, 8
    Tensor<2> qkv; // [T, H_q + 2 * H_kv] 2
    Tensor<2> attn_out; // [T, H_q] 3
    Tensor<2> gate_up; // [T, 2 * I] 6
    Tensor<2> sampling_hidden; // [S, H]
    Tensor<2> logits; // [S, vocab_len]
    CudaDeviceBuffer data;

    static Workspace create(const ModelConfig& config, int max_tokens, int max_seqs);
};

struct ActualWorkspace
{
    Tensor<2> hidden;
    Tensor<2> residual;
    Tensor<2> qkv;
    Tensor<2> attn_out;
    Tensor<2> gate_up;
    Tensor<2> gated;
    Tensor<2> sampling_hidden;
    Tensor<2> logits;

    static ActualWorkspace create(
        const Workspace& workspace,
        int num_tokens,
        int num_sampling_seqs
    );
};

}