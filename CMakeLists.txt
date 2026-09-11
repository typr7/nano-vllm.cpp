cmake_minimum_required(VERSION 3.22)

# Default to Ada Lovelace (compute capability 8.9), while still allowing users
# to select another architecture with -DCMAKE_CUDA_ARCHITECTURES=<value>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

project(cllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

add_library(cllm_core STATIC
    src/batch_buffer.cpp
    src/model_weights.cpp
    src/kv_cache.cpp
    src/executor.cpp
    src/pinned_buffer.cpp
    src/rope_cache.cpp
    src/scheduler.cpp
    src/logger.cpp
    src/kv_cache_manager.cpp
    src/cuda_context.cpp
    src/cuda_device_buffer.cpp
    src/model_config.cpp
    src/engine_core.cpp
    src/model_runner.cpp
    src/sampler.cpp
    src/workspace.cpp

    src/model/causal_lm.cpp

    src/ops/embedding.cu
    src/ops/paged_attention.cu
    src/ops/projection.cu
    src/ops/qk_norm_rope_fused.cu
    src/ops/qk_norm.cu
    src/ops/residual_add.cu
    src/ops/rms_norm.cu
    src/ops/rope.cu
    src/ops/sampler.cu
    src/ops/swiglu.cu
    src/ops/unified_kv_cache_update.cu
)

set_target_properties(cllm_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cllm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cllm_core
    PUBLIC
        msgpack-cxx
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        Threads::Threads
        cppzmq
        nlohmann_json::nlohmann_json
)

# Build the Python extension as _C.so. The Python interpreter is responsible
# for spawning the EngineCore process; a separate C++ executable is not needed.
pybind11_add_module(cllm_pybind MODULE
    src/py_binding.cpp
)

target_link_libraries(cllm_pybind
    PRIVATE
        cllm_core
)

set_target_properties(cllm_pybind PROPERTIES
    OUTPUT_NAME "_C"
)

install(
    TARGETS cllm_pybind
    LIBRARY DESTINATION cllm
)
