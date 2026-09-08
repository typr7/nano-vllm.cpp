#pragma once

#include <type_traits>
#include <cstdint>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "tensor.h"


namespace cllm
{

namespace
{

template <typename T> struct dtype_of;

template<>
struct dtype_of<nv_bfloat16>
{
    static constexpr DataType value = DataType::BF16;
};

}

template <typename T, int DIM>
class TensorView
{
public:
    static_assert(DIM > 0);

    __host__ __device__
    TensorView() noexcept = default;

    __host__ __device__
    TensorView(T* ptr, const int (&shape)[DIM], const int (&stride)[DIM]) noexcept
        : device_ptr_(ptr)
    {
        for (int i = 0; i < DIM; i++) {
            shape_[i] = shape[i];
            stride_[i] = stride[i];
        }
    }

    __host__ __device__
    TensorView(T* ptr, const int (&shape)[DIM]) noexcept
        : device_ptr_(ptr)
    {
        int stride = 1;
        for (int i = DIM - 1; i >= 0; i--) {
            stride_[i] = stride;
            shape_[i] = shape[i];
            stride *= shape[i];
        }
    }

    __host__
    explicit TensorView(const Tensor<DIM>& tensor)
        : device_ptr_(static_cast<T*>(tensor.device_ptr))
    {
        assert(tensor.dtype == dtype_of<std::remove_const_t<T>>::value);
        for (int i = 0; i < DIM; i++) {
            shape_[i] = tensor.shape[i];
            stride_[i] = tensor.stride[i];
        }
    }

    template <typename... Indices>
    __device__
    T& operator()(Indices... indices) const noexcept
    {
        assert(device_ptr_ != nullptr);
        return *(device_ptr_ + offset_of(indices...));
    }

    __host__ __device__
    int shape(int dim) const noexcept
    {
        assert(device_ptr_ != nullptr);
        return shape_[dim];
    }

private:
    template <typename... Indices>
    __device__
    std::int64_t offset_of(Indices... indices) const noexcept
    {
        static_assert(sizeof...(indices) == DIM);
        static_assert((std::is_integral_v<Indices> && ...));

        const int idc[DIM] = {static_cast<int>(indices)...};

        std::int64_t offset = 0;

        #pragma unroll
        for (int i = 0; i < DIM; i++) {
            offset += static_cast<std::int64_t>(idc[i]) * stride_[i];
        }

        return offset;
    }

private:
    T* device_ptr_ = nullptr;
    int shape_[DIM];
    int stride_[DIM];
};

}