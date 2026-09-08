#pragma once

#include "config.h"
#include "model_config.h"
#include "pinned_buffer.h"
#include "cuda_device_buffer.h"
#include "forward_batch.h"
#include "model_runner.h"
#include "cuda_context.h"


namespace cllm
{

class BatchBuffer
{
public:
    static BatchBuffer create(const Config& config, const ModelConfig& model_config);

    ForwardBatch upload(const ModelInput& input, const CudaContext& context);

private:
    CudaDeviceBuffer device_;
    PinnedBuffer pinned_;
};

}