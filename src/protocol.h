#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <msgpack.hpp>


namespace cllm
{

inline constexpr std::string kEngineCoreReady{"READY"};
inline constexpr std::string kEngineCoreInitFailed{"INIT_FAILED"};

class ProtocolError: public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

enum class RequestType: std::uint8_t
{
    kAdd = 0,
    kShutdown = 1,
};

struct EngineCoreRequest
{
    std::string request_id;

    int max_output_tokens = 1024;
    float temperature = 1.f;
    int top_k = 0;
    float top_p = 1.f;

    std::vector<int> token_ids;

    MSGPACK_DEFINE_MAP(request_id, max_output_tokens, temperature, top_k, top_p, token_ids);
};

struct EngineCoreOutput
{
    std::string request_id;
    std::vector<int> token_ids;

    MSGPACK_DEFINE_MAP(request_id, token_ids);
};

using EngineCoreOutputs = std::vector<EngineCoreOutput>;

struct Addresses
{
    std::string handshake_address;
    std::string input_address;
    std::string output_address;
};

enum class EngineCoreShutdownReason
{
    kShutdown,
    kEngineCoreDead
};

inline bool validate_request(const EngineCoreRequest& request, int max_model_len) {
    std::size_t num_tokens = request.token_ids.size();
    return (
        request.max_output_tokens > 0
        && (request.temperature >= 0.f && request.temperature <= 1.f)
        && (request.top_k >= 0 && request.top_k <= 50)
        && (request.top_p > 0.f && request.top_p <= 1.f)
        && (num_tokens > 0 && num_tokens < max_model_len)
    );
}

}
