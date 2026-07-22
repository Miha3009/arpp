#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "../include/patcher.h"

namespace py = pybind11;

PYBIND11_MODULE(patcher, m) {
    m.doc() = "ZSTD-based patched tensor storage with climate anomaly computation";

    py::class_<Context>(m, "Context")
        .def(py::init<const std::string&, size_t>(), 
             "Initialize storage context from directory and thread pool size");

    py::class_<Request>(m, "Request")
        .def(py::init<const std::string&, int, int, const std::string&, int, int, int, int, int>(),
             "Create data request: element, x0, y0, t0, xSize, ySize, tSize, xyStep, tStep")
        .def(py::init<const std::string&, int, int, const std::string&, int, int, int, int, int, int>(),
             "Create data request with tag")
        .def_readwrite("element", &Request::element)
        .def_readwrite("x0", &Request::x0)
        .def_readwrite("y0", &Request::y0)
        .def_readwrite("t0", &Request::t0)
        .def_readwrite("xSize", &Request::xSize)
        .def_readwrite("ySize", &Request::ySize)
        .def_readwrite("tSize", &Request::tSize)
        .def_readwrite("xyStep", &Request::xyStep)
        .def_readwrite("tStep", &Request::tStep)
        .def_readwrite("tag", &Request::tag);

    m.def("train_dict", &train_dict, 
          "Train ZSTD dictionary from training data tensor",
          py::arg("context"), py::arg("data"), py::arg("element"), 
          py::arg("precision"), py::arg("timeInvariant") = false);

    m.def("save", &save, 
          "Save tensor data with dates for given element",
          py::arg("context"), py::arg("data"), py::arg("dates"), 
          py::arg("element"), py::arg("tag") = 0);

    m.def("aggregate", &aggregate, 
          "Aggregate patches into multi-scale pyramid and compute climate anomalies",
          py::arg("context"), py::arg("element"), 
          py::arg("climateStart"), py::arg("climateEnd"));

    m.def("load", &load, 
          "Load data for multiple requests, returns list of 3D or 4D (for ensemble) tensors",
          py::arg("context"), py::arg("requests"));

    m.def("load_climate", &load_climate, 
          "Load climate components for given requests, returns list of 3D tensors",
          py::arg("context"), py::arg("requests"));
}

