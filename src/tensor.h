#pragma once

#include <array>
#include <stdexcept>

#include "data_type.h"


namespace cllm
{

template <int DIM>
struct Tensor
{
    static_assert(DIM > 0);

    void* device_ptr = nullptr;
    DataType dtype = DataType::UNSUPPORTED;
    std::array<int, DIM> shape;
    std::array<int, DIM> stride;

    explicit operator bool() const noexcept { return device_ptr != nullptr; }

    template <typename T>
    T* data() const
    {
        constexpr DataType expected = dtype_of<T>();
        static_assert(expected != DataType::UNSUPPORTED);
        
        if (dtype != expected) {
            throw std::runtime_error("unmatched dtype");
        }

        return static_cast<T*>(device_ptr);
    }
};

template <int DIM>
Tensor<DIM> make_tensor(
    void* device_ptr,
    DataType dtype,
    const std::array<int, DIM>& shape,
    const std::array<int, DIM>& stride
)
{
    return Tensor<DIM>{
        .device_ptr = device_ptr,
        .dtype = dtype,
        .shape = shape,
        .stride = stride
    };
}

template <int DIM>
Tensor<DIM> make_tensor(void* device_ptr, DataType dtype, const std::array<int, DIM>& shape)
{
    std::array<int, DIM> stride;
    int s = 1;
    for (int i = DIM - 1; i >= 0; i--) {
        stride[i] = s;
        s *= shape[i];
    }
    return make_tensor<DIM>(device_ptr, dtype, shape, stride);
}

template <int DIM>
using TensorRef = const Tensor<DIM>&;

}