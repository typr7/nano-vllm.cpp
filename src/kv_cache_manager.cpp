#include <numeric>

#include "kv_cache_manager.h"


namespace cllm
{

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg)
    : num_block_slots_(cfg.block_size),
      num_blocks_(cfg.num_blocks),
      free_blocks_(cfg.num_blocks)
{
    std::iota(free_blocks_.begin(), free_blocks_.end(), 0);
}

KVCacheManager::~KVCacheManager() noexcept
{

}

// TODO: num_occupied_slots_per_block_
bool KVCacheManager::allocate_slots(
    const Request& request, int num_scheduled_tokens
)
{
    int num_remained_slots = num_block_slots_ - request.num_computed_tokens % num_block_slots_;

    int num_blocks_required = (
        num_scheduled_tokens + num_block_slots_ - num_remained_slots - 1
    ) / num_block_slots_;

    if (num_blocks_required > free_blocks_.size()) {
        return false;
    }

    std::vector<int>& assigned_blocks = *req_to_assigned_blocks_[request.id];
    auto begin = free_blocks_.end() - num_blocks_required;
    auto end = free_blocks_.end();

    assigned_blocks.reserve(assigned_blocks.size() + num_blocks_required);
    assigned_blocks.insert(
        assigned_blocks.end(),
        begin,
        end
    );
    free_blocks_.erase(begin, end);

    return true;
}

void KVCacheManager::release_blocks(const std::string& request_id)
{
    std::vector<int>& assigned_blocks = *req_to_assigned_blocks_[request_id];

    free_blocks_.insert(
        free_blocks_.end(),
        assigned_blocks.begin(),
        assigned_blocks.end()
    );

    req_to_assigned_blocks_.erase(request_id);
}

}