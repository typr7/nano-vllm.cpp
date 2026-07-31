#pragma once

#include <string>


namespace cllm
{

struct ModelConfig
{
    int max_model_len;
};

struct Config
{
    std::string model_path;
    float gpu_memory_utilization = 0.8;

    // schedule
    int block_size = 16;
    int max_num_scheduled_tokens;
    int max_chunk_len;

    ModelConfig model_config;
};

}
