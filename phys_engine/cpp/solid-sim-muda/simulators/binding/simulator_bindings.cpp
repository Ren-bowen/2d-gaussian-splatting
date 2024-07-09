#define PY_SSIZE_T_CLEAN
#include "simulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>



namespace py = pybind11;

PYBIND11_MODULE(simulators, m) {
    py::class_<MassSpringSimulator<float, 2>>(m, "MassSpringSimulator2f")
        .def(py::init<std::vector<float>, float, float, std::vector<float>, float, float, int,std::vector<float>, std::vector<float>, std::vector<int>>())
        .def("run", &MassSpringSimulator<float, 2>::run)
        .def("get_x", &MassSpringSimulator<float, 2>::get_x)
        .def("get_v", &MassSpringSimulator<float, 2>::get_v)
        .def("set_v", &MassSpringSimulator<float, 2>::set_v)
        .def("get_covariance", &MassSpringSimulator<float, 2>::get_covariance);

    py::class_<MassSpringSimulator<float, 3>>(m, "MassSpringSimulator3f")
        .def(py::init<std::vector<float>, float, float, std::vector<float>, float, float, int,std::vector<float>, std::vector<float>, std::vector<int>>())
        .def("run", &MassSpringSimulator<float, 3>::run)
        .def("get_x", &MassSpringSimulator<float, 3>::get_x)
        .def("get_v", &MassSpringSimulator<float, 3>::get_v)
        .def("set_v", &MassSpringSimulator<float, 3>::set_v)
        .def("get_covariance", &MassSpringSimulator<float, 3>::get_covariance);

    py::class_<MassSpringSimulator<double, 3>>(m, "MassSpringSimulator3d")
        .def(py::init<std::vector<double>, double, double, std::vector<double>, double, double, int,std::vector<double>, std::vector<double>, std::vector<int>>())
        .def("run", &MassSpringSimulator<double, 3>::run)
        .def("get_x", &MassSpringSimulator<double, 3>::get_x)
        .def("get_v", &MassSpringSimulator<double, 3>::get_v)
        .def("set_v", &MassSpringSimulator<double, 3>::set_v)
        .def("get_covariance", &MassSpringSimulator<double, 3>::get_covariance);
}
    // py::class_<MassSpringSimulator<double, 2>>(m, "MassSpringSimulator2d")
    //     .def(py::init<double, double, double, double, double, double, int>())
    //     .def("run", &MassSpringSimulator<double, 2>::run);

    // py::class_<MassSpringSimulator<float, 3>>(m, "MassSpringSimulator3f")
    //     .def(py::init<float, float, float, float, float, float, int>())
    //     .def("run", &MassSpringSimulator<float, 3>::run);

    // py::class_<MassSpringSimulator<double, 3>>(m, "MassSpringSimulator3d")
    //     .def(py::init<double, double, double, double, double, double, int>())
    //     .def("run", &MassSpringSimulator<double, 3>::run);

