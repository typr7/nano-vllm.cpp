#pragma once

#include <vector>
#include <type_traits>
#include <filesystem>

#include <cuda_bf16.h>

#include "cuda_device_buffer.h"
#include "tensor_view.cuh"
#include "config.h"


namespace cllm
{

template <typename T>
struct LayerWeights
{
    TensorView<T, 1> input_layernorm;
    TensorView<T, 2> qkv_proj;
    TensorView<T, 1> q_norm;
    TensorView<T, 1> k_norm;
    TensorView<T, 2> o_proj;
    TensorView<T, 1> post_attn_layernorm;
    TensorView<T, 2> gate_up_proj;
    TensorView<T, 2> down_proj;
};

template <typename T>
struct ModelWeights
{
    static_assert(std::is_same_v<T, nv_bfloat16>, "supported type(s): bf16");

    TensorView<T, 2> embed_tokens;
    TensorView<T, 1> norm;
    // for tie_word_embeddings == true, it points same buffer as embed_tokens
    TensorView<T, 2> lm_head;
    std::vector<LayerWeights<T>> layers;
    CudaDeviceBuffer data;

    static ModelWeights<T> load_from_safetensors(
        const std::filesystem::path& path,
        const ModelConfig& config
    );
};

}
