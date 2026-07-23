#pragma once

#include <latch>
#include <string>
#include <thread>

#include "config.h"
#include "protocol.h"
#include "queue.hpp"


namespace cllm
{

class EngineCore
{
public:
    EngineCore(const Config& cfg, const EngineCoreAddresses& addresses);
    ~EngineCore() noexcept;

    void run_busy_loop();

    static void run(const Config& cfg, const EngineCoreAddresses& addresses);

private:
    void process_input_socket(
        std::stop_token stop_token,
        const std::string& input_address,
        std::latch& io_ready,
        bool& initialized
    ) noexcept;
    void process_output_socket(
        std::stop_token stop_token,
        const std::string& output_address,
        std::latch& io_ready,
        bool& initialized
    ) noexcept;

    void send_engine_core_dead();

private:
    Queue<InputMessage> input_queue_;
    Queue<OutputMessage> output_queue_;

    std::jthread input_thread_;
    std::jthread output_thread_;
};

}
