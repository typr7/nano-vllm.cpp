#pragma once

#include "model_config.h"
#include "model_weights.h"
#include "cuda_context.h"
#include "forward_batch.h"
#include "kv_cache.h"
#include "workspace.h"
#include "rope_cache.h"


namespace cllm
{

class CausalLM
{
public:
    CausalLM(const ModelConfig& config, ModelWeights&& weights);
    ~CausalLM() noexcept = default;

    void forward(
        const CudaContext& context,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        const ActualWorkspace& workspace
    ) const;

    void compute_logits(
        const CudaContext& context,
        const ForwardBatch& batch,
        const ActualWorkspace& workspace
    ) const;

private:
    void decoder_layer(
        const CudaContext& context,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        int layer,
        const ActualWorkspace& workspace
    ) const;

private:
    ModelConfig config_;
    ModelWeights weights_;
    RopeCache rope_;
};

}