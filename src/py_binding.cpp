#include <pybind11/pybind11.h>

#include "engine_core.h"
#include "protocol.h"


PYBIND11_MODULE(_C, module)
{
    namespace py = pybind11;

    py::class_<cllm::Config>(module, "Config")
        .def(py::init<>())
        .def_readwrite("model_path", &cllm::Config::model_path)
        .def_readwrite("gpu_memory_utilization", &cllm::Config::gpu_memory_utilization)
        .def_readwrite("block_size", &cllm::Config::block_size)
        .def_readwrite("max_num_scheduled_tokens", &cllm::Config::max_num_scheduled_tokens)
        .def_readwrite("max_num_seqs", &cllm::Config::max_num_seqs);

    py::class_<cllm::Addresses>(module, "EngineCoreAddresses")
        .def(py::init<>())
        .def_readwrite("handshake_address", &cllm::Addresses::handshake_address)
        .def_readwrite("input_address", &cllm::Addresses::input_address)
        .def_readwrite("output_address", &cllm::Addresses::output_address);

    py::enum_<cllm::EngineCoreShutdownReason>(module, "EngineCoreShutdownReason")
        .value("kShutdown", cllm::EngineCoreShutdownReason::kShutdown)
        .value("kEngineCoreDead", cllm::EngineCoreShutdownReason::kEngineCoreDead);

    py::class_<cllm::EngineCore>(module, "EngineCore").def_static(
        "run",
        &cllm::EngineCore::run,
        py::call_guard<py::gil_scoped_release>(),
        py::arg("cfg"),
        py::arg("addresses")
    );
}
