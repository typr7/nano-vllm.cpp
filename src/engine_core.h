#pragma once

#include <exception>
#include <latch>
#include <mutex>
#include <string>
#include <thread>

#include "config.h"
#include "protocol.h"
#include "queue.h"


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
        const std::string& input_address
    ) noexcept;
    void process_output_socket(
        std::stop_token stop_token,
        const std::string& output_address
    ) noexcept;

    void report_io_error(std::exception_ptr error) noexcept;
    void rethrow_io_error();
    void send_engine_core_dead();

private:
    Queue<InputMessage> input_queue_;
    Queue<OutputMessage> output_queue_;

    std::mutex io_error_mutex_;
    std::exception_ptr io_error_;
    std::latch io_ready_;

    std::jthread input_thread_;
    std::jthread output_thread_;
};

}
