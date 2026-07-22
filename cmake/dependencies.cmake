find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

# Use CMake's modern FindPython support, which is also understood by pybind11.
set(PYBIND11_FINDPYTHON ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python interpreter: ${Python3_EXECUTABLE}")

# Prefer an already installed pybind11. Fetch a pinned release when it is not
# available so a clean checkout can still be configured directly.
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 was not found locally; fetching v2.13.6")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# cppzmq is a header-only C++ wrapper around libzmq. Prefer system packages,
# but build libzmq locally when its development package is unavailable.
find_package(ZeroMQ CONFIG QUIET)

if(NOT TARGET libzmq AND NOT TARGET libzmq-static)
    message(STATUS "ZeroMQ development package was not found; fetching v4.3.5")
    set(BUILD_SHARED ON CACHE BOOL "Build the shared libzmq library" FORCE)
    set(BUILD_STATIC OFF CACHE BOOL "Do not build the static libzmq library" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Do not build libzmq tests" FORCE)
    set(WITH_PERF_TOOL OFF CACHE BOOL "Do not build libzmq performance tools" FORCE)
    set(WITH_DOCS OFF CACHE BOOL "Do not build libzmq documentation" FORCE)
    set(ENABLE_CPACK OFF CACHE BOOL "Do not configure libzmq packaging" FORCE)
    set(ENABLE_DRAFTS OFF CACHE BOOL "Do not build draft ZeroMQ APIs" FORCE)
    set(ENABLE_WS OFF CACHE BOOL "Do not build WebSocket transport" FORCE)

    FetchContent_Declare(
        libzmq
        GIT_REPOSITORY https://github.com/zeromq/libzmq.git
        GIT_TAG v4.3.5
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(libzmq)
endif()

find_package(cppzmq CONFIG QUIET)

if(NOT TARGET cppzmq)
    message(STATUS "cppzmq was not found locally; fetching v4.11.0")
    set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "Do not build cppzmq tests" FORCE)

    FetchContent_Declare(
        cppzmq
        GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
        GIT_TAG v4.11.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(cppzmq)
endif()

# msgpack-cxx is header-only. Disable optional Boost integration because the
# EngineCore wire format only needs the standard C++17 API.
find_package(msgpack-cxx CONFIG QUIET)

if(NOT TARGET msgpack-cxx)
    message(STATUS "msgpack-cxx was not found locally; fetching cpp-8.0.0")
    set(MSGPACK_CXX11 OFF CACHE BOOL "Do not select C++11 for msgpack" FORCE)
    set(MSGPACK_CXX17 ON CACHE BOOL "Use C++17 for msgpack" FORCE)
    set(MSGPACK_USE_BOOST OFF CACHE BOOL "Build msgpack without Boost" FORCE)
    set(MSGPACK_BUILD_TESTS OFF CACHE BOOL "Do not build msgpack tests" FORCE)
    set(MSGPACK_BUILD_DOCS OFF CACHE BOOL "Do not build msgpack docs" FORCE)
    set(MSGPACK_BUILD_EXAMPLES OFF CACHE BOOL "Do not build msgpack examples" FORCE)

    FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG cpp-8.0.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(msgpack)
endif()
