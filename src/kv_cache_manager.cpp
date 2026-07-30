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
    int num_assigned_blocks = 0;
    auto allocated_iter = req_to_allocated_blocks_.find(request.id);
    if (allocated_iter != req_to_allocated_blocks_.end()) {
        num_assigned_blocks = static_cast<int>(allocated_iter->second->size());
    }

    int num_required_slots = request.num_computed_tokens + num_scheduled_tokens;
    int num_required_blocks = (
        (num_required_slots + num_block_slots_ - 1) / num_block_slots_
    );

    if (num_required_blocks <= num_assigned_blocks) {
        return true;
    }

    int num_blocks_to_allocate = num_required_blocks - num_assigned_blocks;
    if (num_blocks_to_allocate > static_cast<int>(free_blocks_.size())) {
        return false;
    }

    if (allocated_iter == req_to_allocated_blocks_.end()) {
        allocated_iter = req_to_allocated_blocks_.emplace(
            request.id, std::make_shared<std::vector<int>>()
        ).first;
    }

    std::vector<int>& allocated_blocks = *allocated_iter->second;
    auto first_block = free_blocks_.end() - num_blocks_to_allocate;

    allocated_blocks.reserve(allocated_blocks.size() + num_blocks_to_allocate);
    allocated_blocks.insert(
        allocated_blocks.end(), first_block, free_blocks_.end()
    );
    free_blocks_.erase(first_block, free_blocks_.end());

    return true;
}

void KVCacheManager::release_blocks(const std::string& request_id)
{
    if (!req_to_allocated_blocks_.contains(request_id)) {
        throw std::runtime_error("request not existing");
    }

    std::vector<int>& assigned_blocks = *req_to_allocated_blocks_[request_id];

    free_blocks_.insert(
        free_blocks_.end(),
        assigned_blocks.begin(),
        assigned_blocks.end()
    );

    req_to_allocated_blocks_.erase(request_id);
}

}
