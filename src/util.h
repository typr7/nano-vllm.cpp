#pragma once

#include <cstddef>
#include <bit>


namespace cllm
{

template <std::size_t ALIGNMENT>
constexpr std::size_t align_up(std::size_t offset)
{
    static_assert(std::has_single_bit(ALIGNMENT));
    return (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

}