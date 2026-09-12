#include <future>
#include <format>
#include <chrono>

#include <msgpack.hpp>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "engine_core.h"
#include "request.h"
#include "logger.h"


namespace cllm
{

namespace
{

void send_message(const std::string& address, const std::string& message)
{
    zmq::context_t context;
    zmq::socket_t socket(context, zmq::socket_type::push);

    socket.set(zmq::sockopt::linger, 4000);
    socket.connect(address);

    const auto result = socket.send(zmq::buffer(message));
    if (!result) {
        throw std::runtime_error(std::format("failed to send {}", message));
    }
}

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

}

EngineCore::EngineCore(const Config& config)
    : executor_(config),
      model_config_(executor_.model_config()),
      scheduler_(config, model_config_, initialize_kv_cache(executor_, config)),
      shutdown_requested_(false)
{
}

EngineCore::~EngineCore() noexcept
{
    shutdown();
}

KVCacheConfig EngineCore::initialize_kv_cache(Executor& executor, const Config& cfg)
{
    std::size_t byte_size = executor.profile_available_kv_cache_memory(cfg.gpu_memory_utilization);

    // The block layout, including the layer count, is owned by the model runner.
    std::size_t block_byte_size = executor.kv_cache_block_bytes();
    int num_blocks = static_cast<int>(byte_size / block_byte_size);

    executor.allocate_kv_cache(num_blocks);

    return KVCacheConfig{
        .num_block_slots = cfg.block_size,
        .num_blocks = num_blocks
    };
}

void EngineCore::start_io(const Addresses& addresses)
{
    std::promise<void> input_promise;
    std::promise<void> output_promise;

    auto input_ready = input_promise.get_future();
    auto output_ready = output_promise.get_future();

    try {
        input_future_ = std::async(
            std::launch::async,
            &EngineCore::input_thread_main,
            this,
            addresses.input_address,
            std::move(input_promise)
        );

        output_future_ = std::async(
            std::launch::async,
            &EngineCore::output_thread_main,
            this,
            addresses.output_address,
            std::move(output_promise)
        );

        input_ready.get();
        output_ready.get();
    } catch (...) {
        Logger::error("failed to initialize io threads.");
        throw;
    }
}

void EngineCore::shutdown() noexcept
{
    try {
        input_queue_.close();
        if (input_future_.valid()) {
            input_future_.get();
        }
    } catch (...) {}
    try {
        output_queue_.close();
        if (output_future_.valid()) {
            output_future_.get();
        }
    } catch (...) {}
}

void EngineCore::input_thread_main(
    const std::string& address,
    std::promise<void> is_ready
)
{
    std::unique_ptr<zmq::context_t> ctx = nullptr;
    std::unique_ptr<zmq::socket_t> socket = nullptr;

    try {
        ctx = std::make_unique<zmq::context_t>();
        socket = std::make_unique<zmq::socket_t>(*ctx, zmq::socket_type::pull);

        socket->set(zmq::sockopt::rcvhwm, 1024);
        socket->set(zmq::sockopt::rcvtimeo, 100);
        socket->set(zmq::sockopt::linger, 0);
        socket->connect(address);

        is_ready.set_value();
    } catch (...) {
        is_ready.set_exception(std::current_exception());
        return;
    }

    while (!input_queue_.closed()) {
        std::vector<zmq::message_t> frames;
        auto result = zmq::recv_multipart(*socket, std::back_inserter(frames));
        if (!result) {
            continue;
        }

        try {
            const auto& req_type_frame = frames.front();
            if (req_type_frame.size() != sizeof(std::uint8_t)) {
                throw ProtocolError(std::format(
                    "invalid request type byte size: {}", req_type_frame.size()
                ));
            }

            auto req_type = *static_cast<const std::uint8_t*>(req_type_frame.data());
            switch (static_cast<RequestType>(req_type)) {
                case RequestType::kAdd: {
                    if (frames.size() != 2) {
                        throw ProtocolError(std::format(
                            "received ADD request with zmq frame size: {}, expect 2", frames.size()
                        ));
                    }

                    EngineCoreRequest req = decode_request(frames.back());
                    // this is a temp fix and will cause some requests to get stuck
                    // in the frontend process forever.
                    if (!validate_request(req, model_config_.max_model_len)) {
                        Logger::error("invalid request, ignored.");
                        break;
                    }
                    bool ok = input_queue_.push(Request{
                        .id = std::move(req.request_id),
                        .max_output_tokens = static_cast<int>(req.max_output_tokens),
                        .token_ids = std::move(req.token_ids),
                        .sample_params = SampleParams{
                            .temperature = req.temperature,
                            .top_k = static_cast<int>(req.top_k),
                            .top_p = req.top_p
                        }
                    });
                    if (!ok) {
                        return;
                    }

                    break;
                }
                case RequestType::kShutdown: {
                    shutdown_requested_.store(true, std::memory_order_release);

                    return;
                }
                default: {
                    throw ProtocolError("invalid request type");
                }
            }
        } catch (const ProtocolError& e) {
            Logger::error(std::format("invalid request, reason: {}, discarded.", e.what()));
        } catch (...) {
            throw;
        }
    }
}

void EngineCore::output_thread_main(
    const std::string& address,
    std::promise<void> is_ready
)
{
    std::unique_ptr<zmq::context_t> ctx = nullptr;
    std::unique_ptr<zmq::socket_t> socket = nullptr;

    try {
        ctx = std::make_unique<zmq::context_t>();
        socket = std::make_unique<zmq::socket_t>(*ctx, zmq::socket_type::push);

        socket->set(zmq::sockopt::sndhwm, 1024);
        socket->set(zmq::sockopt::sndtimeo, 4000);
        socket->set(zmq::sockopt::immediate, 1);
        socket->set(zmq::sockopt::linger, 4000);
        socket->connect(address);

        is_ready.set_value();
    } catch (...) {
        is_ready.set_exception(std::current_exception());
        return;
    }

    while (auto outputs = output_queue_.pop()) {
        for (const auto& output: *outputs) {
            msgpack::sbuffer payload;
            msgpack::pack(payload, output);
            auto result = socket->send(
                zmq::const_buffer(payload.data(), payload.size()),
                zmq::send_flags::none
            );
            if (!result) {
                throw ProtocolError("failed to send OUTPUT");
            }
        }
    }
}

EngineCoreShutdownReason EngineCore::run(const Config& cfg, const Addresses& addresses)
{
    std::unique_ptr<EngineCore> engine_core = nullptr;

    try {
        if(!validate_config(cfg)) {
            throw std::runtime_error("invalid engine core config");
        }

        engine_core = EngineCore::create(cfg, addresses);

        send_message(addresses.handshake_address, kEngineCoreReady);

        engine_core->run_busy_loop();

        Logger::info("EngineCore shutdown.");

    } catch (const std::exception& e) {
        if (!engine_core) {
            send_message(addresses.handshake_address, kEngineCoreInitFailed);
            Logger::error(std::format(
                "EngineCore failed to initialize due to an exception: {}.",
                e.what()
            ));
        } else {
            // engine_core->send_engine_core_dead();
            Logger::error(std::format("EngineCore terminated due to an exception: {}.", e.what()));
        }

        return EngineCoreShutdownReason::kEngineCoreDead;
    }

    return EngineCoreShutdownReason::kShutdown;
}

std::unique_ptr<EngineCore> EngineCore::create(
    const Config &cfg,
    const Addresses &addresses
)
{
    std::unique_ptr<EngineCore> engine_core(new EngineCore(cfg));

    engine_core->start_io(addresses);

    return engine_core;
}

void EngineCore::run_busy_loop()
{
    while (!shutdown_requested_.load(std::memory_order_acquire)) {
        check_io_threads();

        // process input queue
        std::queue<Request> reqs = input_queue_.drain();
        while (!reqs.empty()) {
            scheduler_.add_request(std::move(reqs.front()));
            reqs.pop();
        }

        // schedule and execute
        std::vector<ScheduledRequest> scheduled = scheduler_.schedule();
        executor_.execute(scheduled);
        std::vector<SampledToken> sampled = executor_.finish();
        EngineCoreOutputs outputs = scheduler_.update(sampled);
        if (!outputs.empty()) {
            output_queue_.push(std::move(outputs));
        }
    }
}

void EngineCore::check_io_threads()
{
    if (input_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        input_future_.get();
    }

    if (output_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        output_future_.get();
    }
}

}
