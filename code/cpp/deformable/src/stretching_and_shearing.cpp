#include "deformable/include/deformable_surface_simulator.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {
namespace deformable {

// For now the material is hard-coded.
// A material whose energy density Psi is quadratic w.r.t. the strain tensor E.
// Psi(F) = 0.5 * E^T C E, where E = 0.5 * vec(F^T F - I).
// mu = 500, lambda = 4000.
static const Matrix4r C = Eigen::Matrix<real, 16, 1>{ 2500., 0., 0., 2000., 0., 500., 0., 0., 0., 0., 500., 0., 2000., 0., 0., 2500. }.reshaped(4, 4);
static const real ComputeEnergyDensityFromStrainTensor(const Matrix2r& E) { return 0.5 * E.reshaped().dot(C * E.reshaped()); }
// The stress tensor P is always symmetric.
static const Matrix2r ComputeStressTensorFromStrainTensor(const Matrix2r& E) { return (C * E.reshaped()).reshaped(2, 2); }

const real Simulator::ComputeStretchingAndShearingEnergy(const Matrix3Xr& position) const {
    real energy = 0;
    for (integer e = 0; e < static_cast<integer>(elements_.cols()); ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix2r E = (F.transpose() * F - Matrix2r::Identity()) / 2;
        energy += ComputeEnergyDensityFromStrainTensor(E);
    }
    // The rest shape area is hard-coded to be 0.005 in this homework.
    return 0.005 * energy;
}

const Matrix3Xr Simulator::ComputeStretchingAndShearingForce(const Matrix3Xr& position) const {
    const integer element_num = static_cast<integer>(elements_.cols());
    std::vector<Matrix3r> gradients(element_num);
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix2r E = (F.transpose() * F - Matrix2r::Identity()) / 2;
        const auto P = ComputeStressTensorFromStrainTensor(E);
        // Derive the gradient dE/dx, where x stands for a 3x3 matrix position(Eigen::all, elements_.col(e)).
        // TODO.
        Eigen::Matrix<real, 6, 9> dFdx;
        dFdx.setZero();
        for (integer i = 0; i < 2; ++i) {
            for (integer j = 0; j < 3; ++j) {
                dFdx(i * 3, j * 3) = D_inv_[e](j, i);
                dFdx(i * 3 + 1, j * 3 + 1) = D_inv_[e](j, i);
                dFdx(i * 3 + 2, j * 3 + 2) = D_inv_[e](j, i);
            }
        }
        Eigen::Matrix<real, 4, 6> dEdF;
        dEdF.setZero();
        dEdF.block(0, 0, 1, 3) = F.col(0).transpose();
        dEdF.block(1, 0, 1, 3) = F.col(1).transpose() / 2;
        dEdF.block(2, 0, 1, 3) = F.col(1).transpose() / 2;
        dEdF.block(1, 3, 1, 3) = F.col(0).transpose() / 2;
        dEdF.block(2, 3, 1, 3) = F.col(0).transpose() / 2;
        dEdF.block(3, 3, 1, 3) = F.col(1).transpose();
        Eigen::Matrix<real, 4, 9> dEdx = dEdF * dFdx;        
        gradients[e] = (P.reshaped(1, 4) * dEdx).reshaped(3, 3) * 0.005;
        /////////////////////////////////
    }

    Matrix3Xr gradient = Matrix3Xr::Zero(3, position.cols());
    for (integer k = 0; k < static_cast<integer>(position.cols()); ++k) {
        for (const auto& tuple : stretching_and_shearing_gradient_map_[k]) {
            const integer e = tuple[0];
            const integer i = tuple[1];
            gradient.col(k) += gradients[e].col(i);
        }
    }

    // Force is negative gradient.
    return -gradient;
}

const SparseMatrixXr Simulator::ComputeStretchingAndShearingHessian(const Matrix3Xr& position) const {
    const integer element_num = static_cast<integer>(elements_.cols());
    std::vector<Matrix9r> hess_nonzeros;
    hess_nonzeros.reserve(element_num);
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix2r E = (F.transpose() * F - Matrix2r::Identity()) / 2;
        const auto P = ComputeStressTensorFromStrainTensor(E);
        // Derive the Hessian d^2E/dx^2, where x stands for a column vector concatenated by the vertices x1, x2, x3.
        // (You do not need to consider the SPD projection issue.)
        // TODO.
        Eigen::Matrix<real, 6, 9> dFdx0;
        dFdx0.setZero();
        for (integer i = 0; i < 2; ++i) {
            for (integer j = 0; j < 3; ++j) {
                dFdx0(i * 3, j * 3) = D_inv_[e](j, i);
                dFdx0(i * 3 + 1, j * 3 + 1) = D_inv_[e](j, i);
                dFdx0(i * 3 + 2, j * 3 + 2) = D_inv_[e](j, i);
            }
        }
        Eigen::Matrix<real, 4, 6> dEdF0;
        dEdF0.setZero();
        dEdF0.block(0, 0, 1, 3) = F.col(0).transpose();
        dEdF0.block(1, 0, 1, 3) = F.col(1).transpose() / 2;
        dEdF0.block(2, 0, 1, 3) = F.col(1).transpose() / 2;
        dEdF0.block(1, 3, 1, 3) = F.col(0).transpose() / 2;
        dEdF0.block(2, 3, 1, 3) = F.col(0).transpose() / 2;
        dEdF0.block(3, 3, 1, 3) = F.col(1).transpose();
        Eigen::Matrix<real, 4, 9> dEdx = dEdF0 * dFdx0;  

        Eigen::Matrix<real, 2, 27> dFdx;
        dFdx.setZero();
        for (integer i = 0; i < 2; ++i) {
            for (integer j = 0; j < 3; ++j) {
                dFdx(i, j * 3) = D_inv_[e](j, i);
                dFdx(i, j * 3 + 9 + 1) = D_inv_[e](j, i);
                dFdx(i, j * 3 + 18 + 2) = D_inv_[e](j, i);
            }
        }
        Eigen::Matrix<real, 4, 54> d_dEdF_dx;
        d_dEdF_dx.setZero();
        d_dEdF_dx.block(0, 0, 1, 27) = dFdx.row(0);
        d_dEdF_dx.block(1, 0, 1, 27) = dFdx.row(1) / 2;
        d_dEdF_dx.block(2, 0, 1, 27) = dFdx.row(1) / 2;
        d_dEdF_dx.block(1, 27, 1, 27) = dFdx.row(0) / 2;
        d_dEdF_dx.block(2, 27, 1, 27) = dFdx.row(0) / 2;
        d_dEdF_dx.block(3, 27, 1, 27) = dFdx.row(1);
        Eigen::Matrix<real, 54, 81> dFdx1;
        dFdx1.setZero();
        for (integer i = 0; i < 2; ++i) {
            for (integer j = 0; j < 3; ++j) {
                for (integer k = 0; k < 9; ++k) {
                    dFdx1(i * 27 + k, j * 27 + k) = D_inv_[e](j, i);
                    dFdx1(i * 27 + 9 + k, j * 27 + 9 + k) = D_inv_[e](j, i);
                    dFdx1(i * 27 + 18 + k, j * 27 + 18 + k) = D_inv_[e](j, i);
                }
            }
        }
        Eigen::Matrix<real, 4, 81> d2Edx2 = d_dEdF_dx * dFdx1;
        Eigen::Matrix<real, 9, 9> hess = ((P.reshaped(1, 4) * d2Edx2).reshaped(9, 9) + dEdx.transpose() * C * dEdx) * 0.005;
        //Eigen::Matrix<real, 9, 9> hess = ((P.reshaped(1, 4) * d2Edx2).reshaped(9, 9)) * 0.005;
        hess_nonzeros.push_back(hess);
        /////////////////////////////////
    }

    SparseMatrixXr ret(stretching_and_shearing_hessian_);
    for (integer k = 0; k < stretching_and_shearing_hessian_nonzero_num_; ++k) {
        real val = 0;
        for (const auto& arr : stretching_and_shearing_hessian_nonzero_map_[k]) {
            const integer e = arr[0];
            const integer i = arr[1];
            const integer j = arr[2];
            val += hess_nonzeros[e](i, j);
        }
        ret.valuePtr()[k] = val;
    }

    return ret;
}

}
}