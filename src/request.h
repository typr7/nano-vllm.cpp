#pragma once

#include <string>
#include <vector>


namespace cllm
{

struct Request
{
    std::string id;

    int num_tokens;

    int num_prompt_tokens;

    // for preempted request that has decoded token,
    // num_prefill_tokens = num_prompt_tokens + num_decoded_tokens_before_preempted
    int num_prefill_tokens;

    // the number of tokens that have been computed, have kv caches stored
    int num_computed_tokens;

    std::vector<int> token_ids;
};

struct RequestData
{
    std::string request_id;

    bool is_prefill;

    int position_offset;

    std::vector<int> tokens_to_compute;
    std::vector<int> allocated_blocks;
};

}