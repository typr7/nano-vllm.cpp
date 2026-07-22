#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <msgpack.hpp>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "engine_core.h"


namespace cllm
{
namespace
{

constexpr int SOCKET_HIGH_WATER_MARK = 1024;
constexpr int SOCKET_TIMEOUT_MS = 100;
constexpr int OUTPUT_LINGER_MS = 4000;

RequestType parse_request_type(const zmq::message_t& frame)
{
    if (frame.size() != sizeof(std::uint8_t)) {
        throw std::runtime_error("Request type frame must contain one byte");
    }

    const auto value = *static_cast<const std::uint8_t*>(frame.data());
    switch (value) {
    case static_cast<std::uint8_t>(RequestType::ADD):
        return RequestType::ADD;
    case static_cast<std::uint8_t>(RequestType::SHUTDOWN):
        return RequestType::SHUTDOWN;
    default:
        throw std::runtime_error("Unknown request type");
    }
}

EngineCoreRequest decode_request(const zmq::message_t& frame)
{
    const auto object = msgpack::unpack(
        static_cast<const char*>(frame.data()),
        frame.size()
    );
    return object.get().as<EngineCoreRequest>();
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

void send_output(zmq::socket_t& socket, const OutputMessage& message)
{
    const auto type = static_cast<std::uint8_t>(message.type);

    if (message.type == OutputType::ENGINE_CORE_DEAD) {
        if (message.payload) {
            throw std::logic_error("ENGINE_CORE_DEAD cannot contain a payload");
        }

        const auto result = socket.send(zmq::buffer(&type, sizeof(type)));
        if (!result) {
            throw std::runtime_error("Failed to send ENGINE_CORE_DEAD");
        }
        return;
    }

    if (!message.payload) {
        throw std::logic_error("OUTPUT must contain a payload");
    }

    msgpack::sbuffer payload;
    msgpack::pack(payload, *message.payload);

    const std::array<zmq::const_buffer, 2> frames{
        zmq::const_buffer(&type, sizeof(type)),
        zmq::const_buffer(payload.data(), payload.size()),
    };
    const auto result = zmq::send_multipart(socket, frames);
    if (!result) {
        throw std::runtime_error("Failed to send EngineCore output");
    }
}

}

EngineCore::EngineCore(
    const Config&,
    const EngineCoreAddresses& addresses
)
    : input_queue_{},
      output_queue_{},
      io_error_mutex_{},
      io_error_{},
      io_ready_{2},
      input_thread_{
          [this, input_address = addresses.input_address](std::stop_token stop_token) {
              process_input_socket(stop_token, input_address);
          }
      },
      output_thread_{
          [this, output_address = addresses.output_address](std::stop_token stop_token) {
              process_output_socket(stop_token, output_address);
          }
      }
{
    io_ready_.wait();
    rethrow_io_error();
}

EngineCore::~EngineCore() noexcept
{
    input_thread_.request_stop();
    if (input_thread_.joinable()) {
        input_thread_.join();
    }
    input_queue_.close();

    output_queue_.close();
    if (output_thread_.joinable()) {
        output_thread_.join();
    }
}

void EngineCore::process_input_socket(
    std::stop_token stop_token,
    const std::string& input_address
) noexcept
{
    bool initialized = false;
    try {
        zmq::context_t context;
        zmq::socket_t socket(context, zmq::socket_type::pull);

        socket.set(zmq::sockopt::rcvhwm, SOCKET_HIGH_WATER_MARK);
        socket.set(zmq::sockopt::rcvtimeo, SOCKET_TIMEOUT_MS);
        socket.set(zmq::sockopt::linger, 0);
        socket.connect(input_address);

        initialized = true;
        io_ready_.count_down();

        while (!stop_token.stop_requested()) {
            std::vector<zmq::message_t> frames;
            const auto result = zmq::recv_multipart(
                socket,
                std::back_inserter(frames)
            );
            if (!result) {
                continue;
            }

            const RequestType type = parse_request_type(frames.front());
            switch (type) {
            case RequestType::ADD:
                if (frames.size() != 2) {
                    throw std::runtime_error("ADD must contain one payload frame");
                }
                input_queue_.push(InputMessage{
                    .type = type,
                    .payload = decode_request(frames[1]),
                });
                break;
            case RequestType::SHUTDOWN:
                if (frames.size() != 1) {
                    throw std::runtime_error("SHUTDOWN cannot contain a payload");
                }
                input_queue_.push(InputMessage{
                    .type = type,
                    .payload = std::nullopt,
                });
                return;
            }
        }
    } catch (...) {
        if (!initialized) {
            io_ready_.count_down();
        }
        report_io_error(std::current_exception());
    }
}

void EngineCore::process_output_socket(
    std::stop_token stop_token,
    const std::string& output_address
) noexcept
{
    bool initialized = false;
    try {
        zmq::context_t context;
        zmq::socket_t socket(context, zmq::socket_type::push);

        socket.set(zmq::sockopt::sndhwm, SOCKET_HIGH_WATER_MARK);
        socket.set(zmq::sockopt::sndtimeo, OUTPUT_LINGER_MS);
        socket.set(zmq::sockopt::immediate, 1);
        socket.set(zmq::sockopt::linger, OUTPUT_LINGER_MS);
        socket.connect(output_address);

        initialized = true;
        io_ready_.count_down();

        while (auto message = output_queue_.pop(stop_token)) {
            send_output(socket, *message);
            if (message->type == OutputType::ENGINE_CORE_DEAD) {
                return;
            }
        }
    } catch (...) {
        if (!initialized) {
            io_ready_.count_down();
        }
        report_io_error(std::current_exception());
    }
}

void EngineCore::report_io_error(std::exception_ptr error) noexcept
{
    {
        const std::lock_guard lock(io_error_mutex_);
        if (!io_error_) {
            io_error_ = error;
        }
    }
    input_queue_.close();
}

void EngineCore::rethrow_io_error()
{
    std::exception_ptr error;
    {
        const std::lock_guard lock(io_error_mutex_);
        error = io_error_;
    }
    if (error) {
        std::rethrow_exception(error);
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
        rethrow_io_error();
        auto message = input_queue_.pop();
        rethrow_io_error();

        if (!message) {
            return;
        }

        switch (message->type) {
        case RequestType::ADD:
            // The scheduler will consume the request in the inference implementation.
            break;
        case RequestType::SHUTDOWN:
            return;
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
