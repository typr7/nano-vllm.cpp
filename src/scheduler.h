#pragma once

#include <list>
#include <vector>

#include "config.h"
#include "kv_cache_manager.h"
#include "protocol.h"
#include "request.h"


namespace cllm
{

// enable: contiunous batching, chunked prefill, kv cache blocks manage, FCFS, preemptive scheduling
// disable: spec decoding, prefix caching, priority scheduling, async scheduling
class Scheduler
{
public:
    Scheduler(const Config& cfg, const KVCacheConfig& kv_cache_config);
    ~Scheduler() noexcept = default;

    std::vector<RequestData> schedule();

    void add_request(const InputMessage& message);

private:
    std::list<Request> running_;
    std::list<Request> waiting_;
    std::list<Request> preempted_;

    KVCacheManager kv_cache_manager_;

    int max_model_len_;
    int max_num_scheduled_tokens_;
    int max_chunk_len_;
};

}