#pragma once

#include <string>
#include <vector>

#include "sample_params.h"


namespace cllm
{

struct Request
{
    std::string id;

    int num_prompt_tokens;

    // for preempted request that has decoded token,
    // num_prefill_tokens = num_prompt_tokens + num_decoded_tokens_before_preempted
    int num_prefill_tokens;

    // the number of tokens that have been computed, have kv caches stored
    int num_computed_tokens;

    int max_output_tokens;

    std::vector<int> token_ids;
    std::vector<int> output_token_ids;

    SampleParams sample_params;
};

struct ScheduledRequest
{
    std::string request_id;

    bool is_prefill;

    // False for every chunk of a chunked prefill except the last one. Only the
    // final chunk produces a token, so only its logits are worth computing.
    bool needs_sampling;

    int position_offset;

    std::vector<int> tokens_to_compute;
    std::vector<int> allocated_blocks;

    SampleParams sample_params;
};

struct SampledToken
{
    std::string request_id;
    int token_id;
    bool eos_token;
};

}
