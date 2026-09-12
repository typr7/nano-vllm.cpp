#pragma once

#include <list>
#include <vector>

#include "config.h"
#include "kv_cache_manager.h"
#include "request.h"
#include "protocol.h"
#include "model_config.h"


namespace cllm
{

using RequestList = std::list<Request>;
using RequestIterator = RequestList::iterator;

// enable: contiunous batching, chunked prefill, kv cache blocks manage, FCFS, preemptive scheduling
// disable: spec decoding, prefix caching, priority scheduling, async scheduling
class Scheduler
{
public:
    Scheduler(
        const Config& cfg,
        const ModelConfig& model_config,
        const KVCacheConfig& kv_cache_config
    );
    ~Scheduler() noexcept = default;

    void add_request(Request request);

    std::vector<ScheduledRequest> schedule();

    EngineCoreOutputs update(const std::vector<SampledToken>& sampled);

private:
    EngineCoreOutput finish_request(RequestIterator request_iter);

    bool reached_token_limit(const Request& request) const noexcept;

private:
    RequestList running_;
    RequestList waiting_;
    RequestList preempted_;

    std::unordered_map<std::string, RequestIterator> id_to_request_;

    KVCacheManager kv_cache_manager_;

    int max_num_seqs_;
    int max_num_scheduled_tokens_;
    int max_model_len_;
};

}