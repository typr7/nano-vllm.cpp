#pragma once

#include <cstdint>
#include <vector>

#include "cuda_context.h"
#include "cuda_device_buffer.h"
#include "pinned_buffer.h"
#include "request.h"
#include "tensor.h"


namespace cllm
{

class Sampler
{
public:
    Sampler(int max_num_seqs, int vocab_size);

    void sample(
        const CudaContext& context,
        TensorRef<2> logits,
        const std::vector<RequestData>& scheduled,
        const std::vector<int>& sampling_request_indices
    );

private:
    CudaDeviceBuffer workspace_;
    CudaDeviceBuffer params_;
    CudaDeviceBuffer token_ids_;
    PinnedBuffer host_params_;
    PinnedBuffer host_token_ids_;
    std::uint64_t seed_;
    std::uint64_t offset_ = 0;
};

}
