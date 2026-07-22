cmake_minimum_required(VERSION 3.22)

project(cllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Ada Lovelace (compute capability 8.9), while still allowing users
# to select another architecture with -DCMAKE_CUDA_ARCHITECTURES=<value>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

add_library(cllm_core STATIC
    src/engine_core.cpp
    src/logger.cpp
)

set_target_properties(cllm_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cllm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cllm_core
    PRIVATE
        CUDA::cudart
        Threads::Threads
        cppzmq
        msgpack-cxx
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
