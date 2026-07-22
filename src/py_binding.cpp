#include <pybind11/pybind11.h>

#include "engine_core.h"


PYBIND11_MODULE(_C, module)
{
    namespace py = pybind11;

    py::class_<cllm::Config>(module, "Config")
        .def(py::init<>())
        .def_readwrite("model_path", &cllm::Config::model_path)
        .def_readwrite("gpu_memory_utilization", &cllm::Config::gpu_memory_utilization);

    py::class_<cllm::EngineCore>(module, "EngineCore").def_static(
        "run",
        &cllm::EngineCore::run,
        py::call_guard<py::gil_scoped_release>(),
        py::arg("cfg")
    );
}