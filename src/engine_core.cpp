#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <msgpack.hpp>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "engine_core.h"
#include "logger.h"


namespace cllm
{
namespace
{

struct IoThreadStartupState
{
    std::latch ready{2};
    std::array<bool, 2> initialized{ false, false };
};

constexpr int SOCKET_HIGH_WATER_MARK = 1024;
constexpr int SOCKET_TIMEOUT_MS = 100;
constexpr int OUTPUT_LINGER_MS = 4000;

EngineCoreRequest decode_request(const zmq::message_t& frame)
{
    try {
        const auto object = msgpack::unpack(
            static_cast<const char*>(frame.data()),
            frame.size()
        );
        return object.get().as<EngineCoreRequest>();
    } catch (...) {
        throw ProtocolError("invalid request payload");
    }
}

void send_ready(const std::string& handshake_address)
{
    zmq::context_t context;
    zmq::socket_t socket(context, zmq::socket_type::push);

    socket.set(zmq::sockopt::linger, OUTPUT_LINGER_MS);
    socket.connect(handshake_address);

    const auto result = socket.send(zmq::buffer(ENGINE_CORE_READY));
    if (!result) {
        throw std::runtime_error("Failed to send EngineCore READY message");
    }
}

}

EngineCore::EngineCore(
    const Config&,
    const EngineCoreAddresses& addresses
)
    : input_queue_{},
      output_queue_{},
      input_thread_{},
      output_thread_{}
{
    auto startup = std::make_shared<IoThreadStartupState>();

    input_thread_ = std::jthread(
        [
            this,
            input_address = addresses.input_address,
            startup
        ](std::stop_token stop_token) {
            process_input_socket(
                stop_token,
                input_address,
                startup->ready,
                startup->initialized[0]
            );
        }
    );
    output_thread_ = std::jthread(
        [
            this,
            output_address = addresses.output_address,
            startup
        ](std::stop_token stop_token) {
            process_output_socket(
                stop_token,
                output_address,
                startup->ready,
                startup->initialized[1]
            );
        }
    );

    startup->ready.wait();
    if (!(startup->initialized[0] && startup->initialized[1])) {
        throw std::runtime_error("failed to initialize io thread");
    }
}

EngineCore::~EngineCore() noexcept
{
    input_thread_.request_stop();
    if (input_thread_.joinable()) {
        input_thread_.join();
    }

    output_queue_.close();
    if (output_thread_.joinable()) {
        output_thread_.join();
    }
    input_queue_.close();
}

void EngineCore::process_input_socket(
    std::stop_token stop_token,
    const std::string& input_address,
    std::latch& io_ready,
    bool& initialized
) noexcept
{
    try {
        zmq::context_t context;
        zmq::socket_t socket(context, zmq::socket_type::pull);

        socket.set(zmq::sockopt::rcvhwm, SOCKET_HIGH_WATER_MARK);
        socket.set(zmq::sockopt::rcvtimeo, SOCKET_TIMEOUT_MS);
        socket.set(zmq::sockopt::linger, 0);
        socket.connect(input_address);

        initialized = true;
        io_ready.count_down();

        while (!stop_token.stop_requested()) {
            std::vector<zmq::message_t> frames;
            const auto result = zmq::recv_multipart(
                socket,
                std::back_inserter(frames)
            );
            if (!result) {
                continue;
            }

            try {
                const auto& req_type_frame = frames.front();
                if (req_type_frame.size() != sizeof(std::uint8_t)) {
                    throw ProtocolError("invalid request type byte size");
                }

                const auto req_type = *static_cast<const std::uint8_t*>(req_type_frame.data());

                switch (req_type) {
                    case static_cast<std::uint8_t>(RequestType::ADD): {
                        if (frames.size() != 2) {
                            throw ProtocolError("");
                        }

                        EngineCoreRequest req = decode_request(frames.back());
                        input_queue_.push(InputMessage{
                            .type = RequestType::ADD,
                            .payload = std::move(req)
                        });
                        break;
                    }
                    case static_cast<std::uint8_t>(RequestType::SHUTDOWN): {
                        input_queue_.push(InputMessage{
                            .type = RequestType::SHUTDOWN,
                            .payload = std::nullopt
                        });
                        return;
                    }
                    default: {
                        throw ProtocolError("invalid request type");
                    }
                }
            } catch (const ProtocolError& e) {
                Logger::error(std::format("invalid request: {}, discarded.", e.what()));
            } catch (...) {
                throw;
            }
        }
    } catch (...) {
        if (!initialized) {
            io_ready.count_down();
            return;
        }
        input_queue_.push_front(InputMessage{
            .type = RequestType::IO_ERROR,
            .payload = std::nullopt
        });
    }
}

void EngineCore::process_output_socket(
    std::stop_token stop_token,
    const std::string& output_address,
    std::latch& io_ready,
    bool& initialized
) noexcept
{
    try {
        zmq::context_t context;
        zmq::socket_t socket(context, zmq::socket_type::push);

        socket.set(zmq::sockopt::sndhwm, SOCKET_HIGH_WATER_MARK);
        socket.set(zmq::sockopt::sndtimeo, OUTPUT_LINGER_MS);
        socket.set(zmq::sockopt::immediate, 1);
        socket.set(zmq::sockopt::linger, OUTPUT_LINGER_MS);
        socket.connect(output_address);

        initialized = true;
        io_ready.count_down();

        while (auto message = output_queue_.pop(stop_token)) {
            OutputType type = message->type;
            switch (type) {
                case OutputType::OUTPUT: {
                    if (!message->payload) {
                        throw ProtocolError("invalid output");
                    }

                    msgpack::sbuffer payload;
                    msgpack::pack(payload, *message->payload);

                    const std::array<zmq::const_buffer, 2> frames{
                        zmq::const_buffer(&type, sizeof(type)),
                        zmq::const_buffer(payload.data(), payload.size())
                    };

                    const auto result = zmq::send_multipart(socket, frames);
                    if (!result) {
                        throw ProtocolError("failed to send OUTPUT");
                    }
                    break;
                }
                case OutputType::ENGINE_CORE_DEAD: {
                    const auto result = socket.send(zmq::const_buffer(&type, sizeof(type)));
                    if (!result) {
                        throw ProtocolError("failed to send ENGINE_CORE_DEAD");
                    }
                    return;
                }
                default: {
                    throw std::runtime_error("invalid output type");
                }
            }
        }
    } catch (...) {
        if (!initialized) {
            io_ready.count_down();
            return;
        }
        input_queue_.push_front(InputMessage{
            .type = RequestType::IO_ERROR,
            .payload = std::nullopt
        });
    }
}

void EngineCore::send_engine_core_dead()
{
    output_queue_.push(OutputMessage{
        .type = OutputType::ENGINE_CORE_DEAD,
        .payload = std::nullopt,
    });
    output_thread_.join();
}

void EngineCore::run_busy_loop()
{
    while (true) {
        auto message = input_queue_.pop();
        if (!message) {
            return;
        }

        switch (message->type) {
            case RequestType::ADD: {
                break;
            }
            case RequestType::SHUTDOWN: {
                return;
            }
            case RequestType::IO_ERROR: {
                throw std::runtime_error("error from io thread");
            }
        }
    }
}

void EngineCore::run(
    const Config& cfg,
    const EngineCoreAddresses& addresses
)
{
    std::unique_ptr<EngineCore> engine;
    try {
        engine = std::make_unique<EngineCore>(cfg, addresses);
        send_ready(addresses.handshake_address);
        engine->run_busy_loop();
    } catch (...) {
        if (engine) {
            engine->send_engine_core_dead();
        }
        throw;
    }
}

}
