#ifndef SIMULATOR
#define SIMULATOR

#include "basic/include/log.hpp"

namespace backend {
namespace deformable {

// Data structures for querying edges.
struct TriangleEdgeInfo {
public:
    real edge_length;
    integer other_triangle;     // Index of the other triangle in elements_. -1 if the edge is at boundary.
};

class Simulator {
public:
    Simulator(const Matrix2Xr& vertices, const Matrix3Xi& elements, const real density, const real bending_stiffness);
    
    void Forward(const real time_step);
    const Matrix3Xr& position() const { return position_; }
    void set_position(const Matrix3Xr& position) { position_ = position; }
    void set_velocity(const Matrix3Xr& velocity) { velocity_ = velocity; }
    void set_external_acceleration(const Matrix3Xr& external_acceleration) { external_acceleration_ = external_acceleration; }

    // Stretching and shearing energy.
    const real ComputeStretchingAndShearingEnergy(const Matrix3Xr& position) const;
    const Matrix3Xr ComputeStretchingAndShearingForce(const Matrix3Xr& position) const;
    const SparseMatrixXr ComputeStretchingAndShearingHessian(
        const Matrix3Xr& position) const;
    const real ComputeBendingEnergy(const Matrix3Xr& position) const;
    const Matrix3Xr ComputeBendingForce(const Matrix3Xr& position) const;
    const SparseMatrixXr ComputeBendingHessian(const Matrix3Xr& position) const;

private:
    // x.
    Matrix3Xr position_;
    // v.
    Matrix3Xr velocity_;
    // g.
    Matrix3Xr external_acceleration_;
    // M.
    SparseMatrixXr int_matrix_;
    // Basis function derivatives.
    std::vector<Eigen::Matrix<real, 3, 2>> D_inv_;
    // Indices.
    const Matrix3Xi elements_;
    // rho.
    const real density_;
    // Coefficient of the bending energy.
    const real bending_stiffness_;
    // Edge data structure for computing bending.
    // triangle_edge_info_[e][i] is the i-th edge of triangle elements_.col(e), which is the vertices in index (elements_(i, e), elements_((i+1)%3, e)).
    std::vector<std::array<TriangleEdgeInfo, 3>> triangle_edge_info_;

    // Accelerating the matrix assembly.
    std::vector<std::vector<std::array<integer, 2>>> stretching_and_shearing_gradient_map_;
    SparseMatrixXr stretching_and_shearing_hessian_;
    integer stretching_and_shearing_hessian_nonzero_num_;
    std::vector<std::vector<std::array<integer, 3>>> stretching_and_shearing_hessian_nonzero_map_;
};

}
}

#endif