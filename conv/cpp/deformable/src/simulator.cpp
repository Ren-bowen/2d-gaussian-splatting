#include "deformable/include/deformable_surface_simulator.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {
namespace deformable {

Simulator::Simulator(const Matrix2Xr& vertices, const Matrix3Xi& elements, const real density,
    const real bending_stiffness) : density_(density), bending_stiffness_(bending_stiffness), elements_(elements) {

    // Initial position and velocity.
    // We assume the rest shape is on the XoY plane.
    const integer dof_num = static_cast<integer>(vertices.cols());
    Matrix3Xr position = Matrix3Xr::Zero(3, dof_num);
    position.row(0) = vertices.row(0); position.row(1) = vertices.row(1);
    position_ = position;
    // Initialize velocity, acceleration, etc.
    velocity_ = Matrix3Xr::Zero(3, dof_num);
    external_acceleration_ = Matrix3Xr::Zero(3, dof_num);

    // Precompute the int matrix.
    std::vector<Eigen::Triplet<real>> int_mat_nonzeros;
    const integer element_num = static_cast<integer>(elements.cols());
    for (integer e = 0; e < element_num; ++e) {
        // For now the area is hard-coded to be 0.005. In general it could be computed from the inputs.
        const real area = 0.005;
        // Hard-coded integration for \Int_{\Omega} w[X]w[X].T dX.
        for (integer i = 0; i < 3; ++i)
            for (integer j = 0; j < 3; ++j)
                int_mat_nonzeros.emplace_back(elements(i, e), elements(j, e), area / (i == j ? 6 : 12));
    }
    int_matrix_ = FromTriplet(dof_num, dof_num, int_mat_nonzeros);

    // Note: different from Hw2, this is constant, so we do not need an outsourced polynomial class.
    for (integer e = 0; e < element_num; ++e) {
        Matrix3r Ds = position(Eigen::all, elements.col(e));
        Ds.row(2) = Vector3r::Ones();
        D_inv_.push_back(Ds.inverse().leftCols(2));
    }

    // Assemble gradient map.
    {
        stretching_and_shearing_gradient_map_.clear();
        stretching_and_shearing_gradient_map_.resize(dof_num);
        for (integer e = 0; e < element_num; ++e) {
            const VectorXi dof_map = elements_.col(e);
            for (integer i = 0; i < 3; ++i) {
                stretching_and_shearing_gradient_map_[dof_map(i)].push_back({ e, i });
            }
        }
    }

    // Assemble the nonzero structures in stretching and shearing energy Hessian.
    {
        const integer dim = 3;
        std::vector<Eigen::Triplet<real>> stretching_and_shearing_hess_nonzeros;
        for (integer e = 0; e < element_num; ++e) {
            const VectorXi dof_map = elements_.col(e);
            for (integer i = 0; i < 3; ++i)
                for (integer j = 0; j < 3; ++j)
                    for (integer di = 0; di < dim; ++di)
                        for (integer dj = 0; dj < dim; ++dj) {
                            const integer row_idx = dof_map(i) * dim + di;
                            const integer col_idx = dof_map(j) * dim + dj;
                            stretching_and_shearing_hess_nonzeros.emplace_back(row_idx, col_idx, static_cast<real>(1));
                        }
        }
        stretching_and_shearing_hessian_ = FromTriplet(dim * dof_num, dim * dof_num, stretching_and_shearing_hess_nonzeros);

        stretching_and_shearing_hessian_nonzero_num_ = static_cast<integer>(stretching_and_shearing_hessian_.nonZeros());
        stretching_and_shearing_hessian_nonzero_map_.clear();
        stretching_and_shearing_hessian_nonzero_map_.resize(stretching_and_shearing_hessian_nonzero_num_);
        for (integer e = 0; e < element_num; ++e) {
            const VectorXi dof_map = elements_.col(e);
            for (integer i = 0; i < 3; ++i)
                for (integer j = 0; j < 3; ++j)
                    for (integer di = 0; di < dim; ++di)
                        for (integer dj = 0; dj < dim; ++dj) {
                            const integer row_idx = dof_map(i) * dim + di;
                            const integer col_idx = dof_map(j) * dim + dj;
                            const integer k = &stretching_and_shearing_hessian_.coeffRef(row_idx, col_idx) - stretching_and_shearing_hessian_.valuePtr();
                            stretching_and_shearing_hessian_nonzero_map_[k].push_back({ e, i * dim + di, j * dim + dj });
                        }
        }
    }

    // Construct triangle_edge_info_.
    triangle_edge_info_.clear();
    triangle_edge_info_.resize(element_num);
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 2, 3>& vertices_2d = vertices(Eigen::all, elements_.col(e));
        for (integer i = 0; i < 3; ++i) {
            auto& info = triangle_edge_info_[e][i];
            info.edge_length = (vertices_2d.col(i) - vertices_2d.col((i + 1) % 3)).norm();
            info.other_triangle = -1;
        }
    }
    std::map<std::pair<integer, integer>, std::pair<integer, integer>> edge_map;
    for (integer e = 0; e < element_num; ++e) {
        for (integer i = 0; i < 3; ++i) {
            const integer idx0 = elements(i, e);
            const integer idx1 = elements((i + 1) % 3, e);
            const std::pair<integer, integer> key = std::make_pair(idx0 < idx1 ? idx0 : idx1,
                idx0 < idx1 ? idx1 : idx0);
            if (edge_map.find(key) == edge_map.end()) {
                // We haven't see this edge before.
                edge_map[key] = std::make_pair(e, i);
            } else {
                // We have seen this edge before.
                const integer other = edge_map[key].first;
                const integer other_edge = edge_map[key].second;
                triangle_edge_info_[e][i].other_triangle = other;
                edge_map.erase(key);
            }
        }
    }
}

void Simulator::Forward(const real time_step) {
    const std::string error_location = "deformable::Simulator::Forward";

    const real h = time_step;
    const real inv_h = 1. / h;
    const integer dof_num = static_cast<integer>(position_.cols());

    const Matrix3Xr& x0 = position_;
    const Matrix3Xr& v0 = velocity_;
    const Matrix3Xr& a = external_acceleration_;

    const Matrix3Xr y = x0 + v0 * h + a * h * h;
    const real half_rho_inv_h2 = density_ * inv_h * inv_h / 2;

    // Functions needed by Newton.
    auto E = [&](const Matrix3Xr& x_next) -> real {
        real energy_kinetic = 0;
        for (integer d = 0; d < 3; ++d) {
            const VectorXr x_next_d(x_next.row(d));
            const VectorXr y_d(y.row(d));
            const VectorXr diff_d = x_next_d - y_d;
            energy_kinetic += diff_d.dot(int_matrix_ * diff_d);
        }
        energy_kinetic *= half_rho_inv_h2;
        const real energy_ss = ComputeStretchingAndShearingEnergy(x_next);
        const real energy_bending = ComputeBendingEnergy(x_next);
        return energy_kinetic + energy_ss + energy_bending;
    };
    // Its gradient.
    auto grad_E = [&](const Matrix3Xr& x_next) -> const VectorXr {
        Matrix3Xr gradient_kinetic = Matrix3Xr::Zero(3, x_next.cols());
        for (integer d = 0; d < 3; ++d) {
            const VectorXr x_next_d(x_next.row(d));
            const VectorXr y_d(y.row(d));
            const VectorXr diff_d = x_next_d - y_d;
            gradient_kinetic.row(d) += RowVectorXr(int_matrix_ * diff_d);
        }
        gradient_kinetic *= density_ * inv_h * inv_h;
        const Matrix3Xr gradient_ss = -ComputeStretchingAndShearingForce(x_next);
        const Matrix3Xr gradient_bending = -ComputeBendingForce(x_next);
        return (gradient_kinetic + gradient_ss + gradient_bending).reshaped();
    };
    auto Hess_E = [&](const Matrix3Xr& x_next) -> const SparseMatrixXr {
        std::vector<Eigen::Triplet<real>> kinetic_nonzeros;
        std::vector<Eigen::Triplet<real>> int_nonzeros = ToTriplet(int_matrix_);
        const real scale = density_ * inv_h * inv_h;
        for (const auto& triplet : int_nonzeros)
            for (integer d = 0; d < 3; ++d) {
                kinetic_nonzeros.push_back(Eigen::Triplet<real>(
                    triplet.row() * 3 + d,
                    triplet.col() * 3 + d,
                    triplet.value() * scale
                ));
            }
        const SparseMatrixXr H_kinetic = FromTriplet(3 * dof_num, 3 * dof_num, kinetic_nonzeros);
        const SparseMatrixXr H_ss = ComputeStretchingAndShearingHessian(x_next);
        const SparseMatrixXr H_bending = ComputeBendingHessian(x_next);
        return H_kinetic + H_ss + H_bending;
    };

    Matrix3Xr xk = x0;
    real Ek = E(xk);
    VectorXr gk = grad_E(xk);
    integer newton_iter = 100;
    while (gk.cwiseAbs().maxCoeff() > 1e-5) {
        Assert(newton_iter > 0, error_location, "Newton iteration failed.");
        Eigen::SimplicialLDLT<SparseMatrixXr> direct_solver(Hess_E(xk));
        const VectorXr pk = direct_solver.solve(-gk);
        // Line search.
        real ls_step = 1.;
        real E_updated = E(xk + pk.reshaped(3, dof_num));
        integer ls_iter = 50;
        while (E_updated > Ek + 0.01 * ls_step * gk.dot(pk)) {
            Assert(ls_iter > 0, error_location, "Line search failed to find sufficient decrease.");
            ls_step /= 2;
            E_updated = E(xk + ls_step * pk.reshaped(3, dof_num));
            --ls_iter;
        }
        xk += ls_step * pk.reshaped(3, dof_num);
        // Exit if no progress could be made.
        if (ls_step * pk.cwiseAbs().maxCoeff() <= 1e-12) break;
        Ek = E_updated;
        gk = grad_E(xk);
        --newton_iter;
    }

    // Update.
    const Matrix3Xr next_position = xk;
    velocity_ = (next_position - position_) * inv_h;
    position_ = next_position;
    // A simple and hard-coded contact handling.
    for (integer i = 0; i < static_cast<integer>(position_.cols()); ++i) {
        Vector3r point = position_.col(i);
        const real point_norm = point.squaredNorm();
        if (point_norm < 1) {
            const real projected_velocity = velocity_.col(i).dot(point);
            if (projected_velocity < 0)
                velocity_.col(i) -= projected_velocity * point / point_norm;
        }
    }
}

}
}