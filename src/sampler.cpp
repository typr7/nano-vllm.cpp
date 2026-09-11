#include <random>

#include "sampler.h"
#include "ops/sampler.h"


namespace cllm
{

Sampler::Sampler(int max_num_seqs, int vocab_size)
    : workspace_(ops::sampler_workspace_size(max_num_seqs, vocab_size)),
      params_(static_cast<std::size_t>(max_num_seqs) * sizeof(SampleParams)),
      token_ids_(static_cast<std::size_t>(max_num_seqs) * sizeof(int)),
      host_params_(params_.size()),
      host_token_ids_(token_ids_.size()),
      seed_(std::random_device{}())
{
}

void Sampler::sample(
    const CudaContext& context,
    TensorRef<2> logits,
    const std::vector<RequestData>& scheduled,
    const std::vector<int>& sampling_request_indices
)
{
    const int num_sampling_reqs = static_cast<int>(sampling_request_indices.size());
    if (num_sampling_reqs == 0) {
        return;
    }

    auto* params = host_params_.data<SampleParams>();
    for (int i = 0; i < num_sampling_reqs; i++) {
        params[i] = scheduled[sampling_request_indices[i]].sample_params;
    }
    params_.upload_async(
        params,
        num_sampling_reqs * sizeof(SampleParams),
        context.stream()
    );

    ops::sample(
        logits,
        params_.data<SampleParams>(),
        token_ids_.data<int>(),
        workspace_.data(),
        seed_,
        offset_,
        context.stream()
    );
    offset_ += num_sampling_reqs;

    token_ids_.download_async(
        host_token_ids_.data(),
        num_sampling_reqs * sizeof(int),
        context.stream()
    );

    /*
    const auto* token_ids = host_token_ids_.data<int>();
    std::vector<SamplerOutput> output;
    output.reserve(num_sampling_reqs);
    for (int i = 0; i < num_sampling_reqs; i++) {
        output.push_back(SamplerOutput{
            .request_id = scheduled[sampling_request_indices[i]].request_id,
            .token_id = token_ids[i]
        });
    }
    return output;
    */
}

}
