#pragma once

#include <memory>
#include <future>
#include <memory>

#include "executor.h"
#include "scheduler.h"
#include "queue.hpp"
#include "protocol.h"
#include "config.h"
#include "kv_cache_manager.h"


namespace cllm
{

class EngineCore
{
public:
    ~EngineCore() noexcept;

    static void run(const Config& cfg, const Addresses& addresses);
    
    static std::unique_ptr<EngineCore> create(const Config& cfg, const Addresses& addresses);

    void shutdown() noexcept;

private:
    EngineCore(const Config& cfg);

    EngineCore(const EngineCore&) = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    void run_busy_loop();

    // io
    void send_engine_core_dead() noexcept;

    void input_thread_main(const std::string& address, std::promise<void> is_ready);
    void output_thread_main(const std::string& address, std::promise<void> is_ready);

    void start_io(const Addresses& addresses);

    void check_io_threads();

    // kv cache
    static KVCacheConfig initialize_kv_cache(Executor& executor, const Config& cfg);

private:
    std::atomic<bool> shutdown_requested_;

    // io
    Queue<Request> input_queue_;
    Queue<OutputMessage> output_queue_;

    std::future<void> input_future_;
    std::future<void> output_future_;

    Executor executor_;

    Scheduler scheduler_;
};

}