#pragma once

#include <string>


namespace cllm
{

struct Config
{
    std::string model_path;
    float gpu_memory_utilization;
};

}
