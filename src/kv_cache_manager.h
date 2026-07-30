#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "request.h"


namespace cllm
{

struct KVCacheConfig
{
    int block_size;
    int num_blocks;
};

struct KVCacheBlock
{
    int block_id;
};

class KVCacheManager
{
public:
    KVCacheManager(const KVCacheConfig& cfg);
    ~KVCacheManager() noexcept;

    bool allocate_slots(const Request& request, int num_scheduled_tokens);

    void release_blocks(const std::string& request_id);

private:
    int num_block_slots_;

    int num_blocks_;

    std::vector<int> free_blocks_;

    std::unordered_map<std::string, std::shared_ptr<std::vector<int>>> req_to_allocated_blocks_;
};

}