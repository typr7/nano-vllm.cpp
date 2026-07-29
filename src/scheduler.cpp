#include "scheduler.h"


namespace cllm
{

Scheduler::Scheduler(const Config& config, const KVCacheConfig& kv_cache_config)
    : kv_cache_manager_(kv_cache_config),
      max_num_scheduled_tokens_(config.max_num_scheduled_tokens),
      max_chunk_len_(config.max_chunk_len)
{
}

Scheduler::~Scheduler() noexcept
{
}

std::vector<RequestData> Scheduler::schedule()
{
    int token_budget = max_num_scheduled_tokens_;

    // pass 1: running
    auto req_iter = running_.begin();
    while (req_iter != running_.end() && token_budget > 0) {
        const Request& cur_req = *req_iter;
        bool prefill = cur_req.num_computed_tokens < cur_req.prompt_tokens.size();

        int num_scheduled_tokens = cur_req.num_tokens - cur_req.num_computed_tokens;
        num_scheduled_tokens = std::min(
            {num_scheduled_tokens, max_chunk_len_, token_budget}
        );

        // try assgin kv cache blocks to new tokens
        bool allocated = false;
        while (true) {
            if (kv_cache_manager_.allocate_slots(cur_req, num_scheduled_tokens)) {
                allocated = true;
                break;
            }

            // preempt a request from back of running queue
            Request& req_to_preempt = running_.back();
            bool is_cur_req = req_to_preempt.id == cur_req.id;

            kv_cache_manager_.release_blocks(req_to_preempt.id);
            req_to_preempt.num_computed_tokens = 0;

            preempted_.push(std::move(req_to_preempt));

            if (is_cur_req) {
                break;
            }
        }

        if (!allocated) {
            break;
        }
        
        // TODO
    }
}

}