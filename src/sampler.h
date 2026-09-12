#pragma once

#include <cstdint>

#include "cuda_context.h"
#include "cuda_device_buffer.h"
#include "forward_batch.h"
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
        const ForwardBatch& batch
    );

private:
    CudaDeviceBuffer workspace_;
    std::uint64_t seed_;
    std::uint64_t offset_ = 0;
};

}
