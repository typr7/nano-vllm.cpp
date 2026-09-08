#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <msgpack.hpp>


namespace cllm
{

inline constexpr std::string ENGINE_CORE_READY{"READY"};
inline constexpr std::string ENGINE_CORE_INIT_FAILED{"INIT_FAILED"};

class ProtocolError: public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

enum class RequestType: std::uint8_t
{
    ADD = 0,
    SHUTDOWN = 1,
};

enum class OutputType: std::uint8_t
{
    OUTPUT = 0,
    ENGINE_CORE_DEAD = 1,
};

struct EngineCoreRequest
{
    std::string request_id;

    std::uint32_t max_output_tokens = 1024;
    float temperature = 1.f;
    std::uint32_t top_k = 0;
    float top_p = 1.f;

    std::vector<std::int32_t> token_ids;

    MSGPACK_DEFINE_MAP(request_id, max_output_tokens, temperature, top_k, top_p, token_ids);
};

struct EngineCoreOutput
{
    std::string request_id;
    std::vector<std::int32_t> token_ids;

    MSGPACK_DEFINE_MAP(request_id, token_ids);
};

struct Addresses
{
    std::string handshake_address;
    std::string input_address;
    std::string output_address;
};

struct OutputMessage
{
    OutputType type;
    std::optional<EngineCoreOutput> payload;
};

}
