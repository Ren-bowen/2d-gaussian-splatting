#pragma once

#include <vector>
#include <cmath>
#include "square_mesh.h"
#include <iostream>
template <typename T, int dim>
class MassSpringSimulator
{
public:
    MassSpringSimulator();
    ~MassSpringSimulator();
    MassSpringSimulator(MassSpringSimulator &&rhs);
    MassSpringSimulator &operator=(MassSpringSimulator &&rhs);
    MassSpringSimulator(std::vector<T> M, T side_len, T initial_stretch, std::vector<T> K, T h, T tol, int n_seg, std::vector<T> initial_length, std::vector<T> covariance, std::vector<int> elements);
    void run();
    std::vector<T> get_x();
    std::vector<T> get_v();
    std::vector<T> get_covariance();
    void set_v(const std::vector<T> v_new);
private:
    // The implementation details of the VecAdder class are placed in the implementation class declared here.
    struct Impl;
    // The private pointer to the implementation class Impl
    std::unique_ptr<Impl> pimpl_;
};
