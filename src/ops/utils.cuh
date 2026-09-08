#pragma once

#include <cmath>
#include <cstdint>

#include <vector_types.h>
#include <cuda_bf16.h>


namespace cllm::ops
{

inline constexpr uint32_t kNumThreadsPerWarp = 32;

inline constexpr uint32_t kNumBf16sPerVector = sizeof(uint4) / sizeof(nv_bfloat16);

union BF16x8
{
    uint4 vec;
    nv_bfloat16 bf16x8[8];
};

template <uint32_t kWidth = 32>
requires (2 <= kWidth && kWidth <= 32 && std::has_single_bit(kWidth))
__device__ __forceinline__
float warp_reduce_sum(float val)
{
    #pragma unroll
    for (uint32_t i = (kWidth >> 1); i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i, kWidth);
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_max(float val)
{
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__
uint32_t pack_bf16x2(float low, float high)
{
    return static_cast<uint32_t>(__bfloat16_as_ushort(low))
           | (static_cast<uint32_t>(__bfloat16_as_ushort(high)) << 16);
}

template <typename ToType, typename FromType>
ToType& as(FromType* p)
{
    return *reinterpret_cast<ToType* const>(p);
}

}