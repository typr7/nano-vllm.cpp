#include <cassert>

#include <cublas_v2.h>

#include "projection.h"
#include "cuda_utils.h"


namespace cllm::ops
{

// [M, K], [N, K], [M, N]
void projection(
    TensorRef<2> input,
    TensorRef<2> weights,
    TensorRef<2> output,
    cublasHandle_t handle
)
{
    constexpr float ALPHA_ONE = 1.f;
    constexpr float BETA_ZERO = 0.f;

    assert(input);
    assert(weights);
    assert(output);
    assert(input.shape[0] == output.shape[0]);
    assert(input.shape[1] == weights.shape[1]);
    assert(weights.shape[0] == output.shape[1]);
    assert(input.dtype == weights.dtype);
    assert(input.dtype == output.dtype);

    const int M = input.shape[0];
    const int N = weights.shape[0];
    const int K = input.shape[1];

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &ALPHA_ONE,
        weights.device_ptr, CUDA_R_16BF, weights.stride[0],
        input.device_ptr, CUDA_R_16BF, input.stride[0],
        &BETA_ZERO,
        output.device_ptr, CUDA_R_16BF, output.stride[0],
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

}