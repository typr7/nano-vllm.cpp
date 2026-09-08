#include <cassert>
#include <algorithm>
#include <iterator>

#include "scheduler.h"


namespace cllm
{

Scheduler::Scheduler(const Config& config, const KVCacheConfig& kv_cache_config)
    : kv_cache_manager_(kv_cache_config),
      // max_model_len_(config.model_config.max_model_len),
      max_num_scheduled_tokens_(config.max_num_scheduled_tokens),
      max_chunk_len_(config.max_chunk_len)
{
}

void Scheduler::add_request(Request request)
{
    int num_tokens = static_cast<int>(request.token_ids.size());
    request.num_prompt_tokens = num_tokens;
    request.num_prefill_tokens = num_tokens;
    request.num_computed_tokens = 0;
    waiting_.push_back(std::move(request));
}

std::vector<RequestData> Scheduler::schedule()
{
    std::vector<RequestData> scheduled;

    int token_budget = max_num_scheduled_tokens_;

    // pass 1: running
    std::size_t num_preempted_before = preempted_.size();
    auto cur_iter = running_.begin();
    while (cur_iter != running_.end() && token_budget > 0) {
        int num_tokens = static_cast<int>(cur_iter->token_ids.size());
        int num_scheduled_tokens = num_tokens - cur_iter->num_computed_tokens;
        num_scheduled_tokens = std::min(
            {num_scheduled_tokens, max_chunk_len_, token_budget}
        );
        assert(num_scheduled_tokens > 0);

        // try assgin kv cache blocks to new tokens
        bool allocated = false;
        while (true) {
            if (kv_cache_manager_.allocate_slots(*cur_iter, num_scheduled_tokens)) {
                allocated = true;
                break;
            }

            auto preempted_iter = std::prev(running_.end());

            kv_cache_manager_.release_blocks(preempted_iter->id);
            preempted_iter->num_computed_tokens = 0;

            preempted_.splice(preempted_.begin(), running_, preempted_iter);

            if (preempted_iter->id == cur_iter->id) {
                break;
            }
        }

        if (!allocated) {
            break;
        }

        token_budget -= num_scheduled_tokens;

        const auto compute_begin = cur_iter->token_ids.begin() + cur_iter->num_computed_tokens;
        const int num_computed_after = cur_iter->num_computed_tokens + num_scheduled_tokens;
        scheduled.push_back(RequestData{
            .request_id = cur_iter->id,
            .is_prefill = cur_iter->num_computed_tokens < cur_iter->num_prefill_tokens,
            .needs_sampling = num_computed_after >= cur_iter->num_prefill_tokens,
            .position_offset = cur_iter->num_computed_tokens,
            .tokens_to_compute = std::vector<int>(
                compute_begin,
                compute_begin + num_scheduled_tokens
            ),
            .allocated_blocks = kv_cache_manager_.get_allocated_blocks(cur_iter->id),
            .sample_params = cur_iter->sample_params
        });

        cur_iter++;
    }

    if (preempted_.size() == num_preempted_before) {
        // pass 2: preempted and waiting
        while ((!preempted_.empty() || !waiting_.empty()) && token_budget > 0) {
            auto& request_queue = preempted_.empty() ? waiting_ : preempted_;
            auto cur_iter = request_queue.begin();

            int num_tokens = static_cast<int>(cur_iter->token_ids.size());
            int num_scheduled_tokens = std::min(
                {num_tokens, max_chunk_len_, token_budget}
            );
            assert(num_scheduled_tokens > 0);

            if (!kv_cache_manager_.allocate_slots(*cur_iter, num_scheduled_tokens)) {
                break;
            }

            cur_iter->num_prefill_tokens = static_cast<int>(cur_iter->token_ids.size());
            token_budget -= num_scheduled_tokens;

            scheduled.push_back(RequestData{
                .request_id = cur_iter->id,
                .is_prefill = true,
                .needs_sampling = num_scheduled_tokens >= cur_iter->num_prefill_tokens,
                .position_offset = 0,
                .tokens_to_compute = std::vector<int>(
                    cur_iter->token_ids.begin(),
                    cur_iter->token_ids.begin() + num_scheduled_tokens
                ),
                .allocated_blocks = kv_cache_manager_.get_allocated_blocks(cur_iter->id),
                .sample_params = cur_iter->sample_params
            });

            running_.splice(running_.end(), request_queue, cur_iter);
        }
    }

    return scheduled;
}

}