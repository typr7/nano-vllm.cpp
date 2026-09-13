#pragma once

#include "config.h"
#include "protocol.h"


namespace cllm
{

// export to python via pybind
EngineCoreShutdownReason run_engine_core(const Config& config, const Addresses& addresses);

}
