#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "deformable/include/deformable_surface_simulator.hpp"

PYBIND11_MODULE(backend, m) {
    pybind11::class_<backend::deformable::Simulator>(m, "Simulator")
        .def(pybind11::init<const backend::Matrix2Xr&, const backend::Matrix3Xi&, const backend::real, const backend::real>(),
            pybind11::arg("vertices"), pybind11::arg("elements"), pybind11::arg("density"), pybind11::arg("bending_stiffness"))
        .def("Forward", &backend::deformable::Simulator::Forward, pybind11::arg("time_step"))
        .def("position", &backend::deformable::Simulator::position)
        .def("set_position", &backend::deformable::Simulator::set_position)
        .def("set_velocity", &backend::deformable::Simulator::set_velocity)
        .def("set_external_acceleration", &backend::deformable::Simulator::set_external_acceleration)
        .def("ComputeStretchingAndShearingEnergy", &backend::deformable::Simulator::ComputeStretchingAndShearingEnergy)
        .def("ComputeStretchingAndShearingForce", &backend::deformable::Simulator::ComputeStretchingAndShearingForce)
        .def("ComputeStretchingAndShearingHessian", &backend::deformable::Simulator::ComputeStretchingAndShearingHessian)
        .def("ComputeBendingEnergy", &backend::deformable::Simulator::ComputeBendingEnergy)
        .def("ComputeBendingForce", &backend::deformable::Simulator::ComputeBendingForce)
        .def("ComputeBendingHessian", &backend::deformable::Simulator::ComputeBendingHessian);
}