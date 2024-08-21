#include "deformable/include/deformable_surface_simulator.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {
namespace deformable {

static const real ComputeDihedralAngleFromNonUnitNormal(const Vector3r& normal, const Vector3r& other_normal) {
    const real sin_angle = normal.cross(other_normal).norm();
    const real cos_angle = normal.dot(other_normal);
    const real angle = std::atan2(sin_angle, cos_angle);
    return angle;
}

static const Vector3r ComputeNormal(const Matrix3r& vertices) {
    // This is the normal direction vector that is not normalized.
    // You may assume that in this homework the area of a triangle does not shrink below 1e-5,
    // therefore the normal direction (a x b)/||a x b|| does not suffer from numerical issues.
    return (vertices.col(1) - vertices.col(0)).cross(vertices.col(2) - vertices.col(1));
}

static const Matrix3r CrossProductMatrix(const Vector3r& v) {
    Matrix3r ret;
    ret << 0, -v(2), v(1),
           v(2), 0, -v(0),
           -v(1), v(0), 0;
    return ret;
}

static const Matrix3Xr ComputeNormalGradient(const Matrix3r& vertices) {
    Matrix3Xr grad = Matrix3Xr::Zero(3, 9);
    grad.block(0, 0, 3, 3) = CrossProductMatrix(vertices.col(2) - vertices.col(1));
    grad.block(0, 3, 3, 3) = CrossProductMatrix(vertices.col(0) - vertices.col(2));
    grad.block(0, 6, 3, 3) = CrossProductMatrix(vertices.col(1) - vertices.col(0));
    return grad;
}

static const std::pair<Vector3r, Vector3r> ComputeAngleGradient(const Vector3r& normal, const Vector3r& other_normal) {
    const Vector3r n_cross = normal.cross(other_normal);
    const real sin_angle = n_cross.norm();
    const real cos_angle = normal.dot(other_normal);
    Vector3r unit_n_cross = Vector3r::Zero();
    if (sin_angle > 1e-7 * normal.norm() * other_normal.norm()) {
        unit_n_cross = n_cross / sin_angle;
    }
    const Vector3r sin0 = other_normal.cross(unit_n_cross);
    const Vector3r sin1 = -normal.cross(unit_n_cross);
    const Vector3r cos0 = other_normal;
    const Vector3r cos1 = normal;
    const real angle = std::atan2(sin_angle, cos_angle);
    const real dangle_dsin = cos_angle / (cos_angle * cos_angle + sin_angle * sin_angle);
    const real dangle_dcos = -sin_angle / (cos_angle * cos_angle + sin_angle * sin_angle);
    const Vector3r grad0 = dangle_dsin * sin0 + dangle_dcos * cos0;
    const Vector3r grad1 = dangle_dsin * sin1 + dangle_dcos * cos1;
    return { grad0, grad1 };
}

static const std::array<Matrix3Xr, 9> ComputeNormalHessian(const Matrix3r& vertices) {
    std::array<Matrix3Xr, 9> hess;
    hess.fill(Matrix3Xr::Zero(3, 9));
    for (integer i = 0; i < 3; i++) {
        for (integer j = 0; j < 3; j++) {
            hess[((i + 1) % 3) * 3 + j].block(0, 3 * i, 3, 3) -= CrossProductMatrix(Vector3r::Unit(j));
            hess[((i + 2) % 3) * 3 + j].block(0, 3 * i, 3, 3) += CrossProductMatrix(Vector3r::Unit(j));
        }
    }
    return hess;
}

static const std::array<Matrix3r, 4> ComputeAngleHessian(const Vector3r& normal, const Vector3r& other_normal) {
    std::array<Matrix3r, 4> hess;

    const Vector3r n_cross = normal.cross(other_normal);
    const real sin_angle = normal.cross(other_normal).norm();
    const real cos_angle = normal.dot(other_normal);
    Vector3r unit_n_cross = Vector3r::Zero();
    Matrix3r dunit_n_cross0 = Matrix3r::Zero();
    Matrix3r dunit_n_cross1 = Matrix3r::Zero();    
    if (sin_angle > 1e-8 * normal.norm() * other_normal.norm()) {
        unit_n_cross = n_cross / sin_angle;
        dunit_n_cross0 = -CrossProductMatrix(other_normal) / sin_angle + (unit_n_cross * unit_n_cross.transpose()) * CrossProductMatrix(other_normal) / sin_angle;
        dunit_n_cross1 = CrossProductMatrix(normal) / sin_angle - (unit_n_cross * unit_n_cross.transpose()) * CrossProductMatrix(normal) / sin_angle;
    }

    const Vector3r dsin0 = other_normal.cross(unit_n_cross);
    const Vector3r dsin1 = -normal.cross(unit_n_cross);
    const Matrix3r d2sin00 = CrossProductMatrix(other_normal) * dunit_n_cross0;
    const Matrix3r d2sin01 = CrossProductMatrix(other_normal) * dunit_n_cross1 - CrossProductMatrix(unit_n_cross);
    const Matrix3r d2sin10 = -CrossProductMatrix(normal) * dunit_n_cross0 + CrossProductMatrix(unit_n_cross);

    const Matrix3r d2sin11 = -CrossProductMatrix(normal) * dunit_n_cross1;

    const real sqr = cos_angle * cos_angle + sin_angle * sin_angle;     // = other_normal.dot(other_normal) * normal.dot(normal);  
    const Vector3r dcos0 = other_normal;
    const Vector3r dcos1 = normal;
    
    const Vector3r dsqr0 = 2 * cos_angle * dcos0 + 2 * sin_angle * dsin0;
    const Vector3r dsqr1 = 2 * cos_angle * dcos1 + 2 * sin_angle * dsin1;    
    //const Vector3r dsqr0 = 2 * other_normal.dot(other_normal) * dcos1;
    //const Vector3r dsqr1 = 2 * normal.dot(normal) * dcos0;
    const real dangle_dsin = cos_angle / sqr;
    const real dangle_dcos = -sin_angle / sqr;

    const Vector3r d2angle_dsin0 = (dcos0 * sqr - cos_angle * dsqr0) / (sqr * sqr);
    const Vector3r d2angle_dsin1 = (dcos1 * sqr - cos_angle * dsqr1) / (sqr * sqr);
    const Vector3r d2angle_dcos0 = -(dsin0 * sqr - sin_angle * dsqr0) / (sqr * sqr);
    const Vector3r d2angle_dcos1 = -(dsin1 * sqr - sin_angle * dsqr1) / (sqr * sqr);

    const Matrix3r d2angle00 = dangle_dsin * d2sin00 + dsin0 * d2angle_dsin0.transpose() + dcos0 * d2angle_dcos0.transpose();
    const Matrix3r d2angle01 = dangle_dsin * d2sin01 + dsin0 * d2angle_dsin1.transpose() + dcos0 * d2angle_dcos1.transpose() + dangle_dcos * Matrix3r::Identity();
    const Matrix3r d2angle10 = dangle_dsin * d2sin10 + dsin1 * d2angle_dsin0.transpose() + dcos1 * d2angle_dcos0.transpose() + dangle_dcos * Matrix3r::Identity();
    const Matrix3r d2angle11 = dangle_dsin * d2sin11 + dsin1 * d2angle_dsin1.transpose() + dcos1 * d2angle_dcos1.transpose();

    hess[0] = d2angle00;
    hess[1] = d2angle01;
    hess[2] = d2angle10;
    hess[3] = d2angle11;

    return hess;
}

// In this homework, the rest shape area is fixed and hard-coded to be 0.005. In general it could be computed from the inputs. 
const real Simulator::ComputeBendingEnergy(const Matrix3Xr& position) const {
    // Loop over all edges.
    const integer element_num = static_cast<integer>(elements_.cols());
    real energy = 0;
    for (integer e = 0; e < element_num; ++e) {
        // Compute normal.
        const Vector3r normal = ComputeNormal(position(Eigen::all, elements_.col(e)));
        for (integer i = 0; i < 3; ++i) {
            const TriangleEdgeInfo& info = triangle_edge_info_[e][i];
            // We only care about internal edges and only computes each edge once.
            // Update April 7th: we stored only each edge info once, so the condition e > info.other_triangle is removed. 
            if (info.other_triangle == -1) continue;
            const Vector3r other_normal = ComputeNormal(position(Eigen::all, elements_.col(info.other_triangle)));
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;
            // TODO.
            energy += angle * angle * rest_shape_edge_length * rest_shape_edge_length / diamond_area;

            //energy += angle * angle / 0.1;
            //std::cout<< "angle: " << angle << " rest_shape_edge_length: " << rest_shape_edge_length << " diamond_area: " << diamond_area << " energy: " << energy << std::endl;
            //std::cout<<angle * angle * rest_shape_edge_length * rest_shape_edge_length / (diamond_area / 2)<<std::endl;
            /////////////////////////////////////////////
        }
    }
    return bending_stiffness_ * energy;
}

const Matrix3Xr Simulator::ComputeBendingForce(const Matrix3Xr& position) const {
    // TODO.
    const integer element_num = static_cast<integer>(elements_.cols());
    Matrix3Xr dEdx = Matrix3Xr::Zero(3, position.cols());
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix3r vs = position(Eigen::all, elements_.col(e));
        const Vector3r normal = ComputeNormal(vs);
        const Matrix3Xr normal_gradient = ComputeNormalGradient(vs);
        for (integer i = 0; i < 3; ++i) {
            Vector3r X0 = position(Eigen::all, elements_(i, e));
            Vector3r X1 = position(Eigen::all, elements_((i + 1) % 3, e));
            const TriangleEdgeInfo& info = triangle_edge_info_[e][i];  
            if (info.other_triangle == -1) continue;
            const Matrix3r other_vs = position(Eigen::all, elements_.col(info.other_triangle));
            const Vector3r other_normal = ComputeNormal(other_vs);
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;

            const Matrix3Xr other_normal_gradient = ComputeNormalGradient(other_vs);
            const auto [dangle_dnormal, dangle_dother_normal] = ComputeAngleGradient(normal, other_normal);
            const Matrix3r dangle_dv = (dangle_dnormal.transpose() * normal_gradient).reshaped(3, 3);
            const Matrix3r dangle_dother_v = (dangle_dother_normal.transpose() * other_normal_gradient).reshaped(3, 3);

            const real coefficient = 2 * (angle * rest_shape_edge_length) * rest_shape_edge_length / diamond_area;

            for (integer j = 0; j < 3; j++) {
                dEdx.col(elements_(j, e)) += coefficient * dangle_dv.col(j);
                dEdx.col(elements_(j, info.other_triangle)) += coefficient * dangle_dother_v.col(j);
            }
        }
    }
    return -bending_stiffness_ * dEdx;
    /////////////////////////////////////
}

const SparseMatrixXr Simulator::ComputeBendingHessian(const Matrix3Xr& position) const {
    // TODO.
    const integer hess_size = static_cast<integer>(position.cols()) * 3;
    MatrixXr hess(hess_size, hess_size);
    hess.setZero();
    const integer element_num = static_cast<integer>(elements_.cols());
    std::vector<Matrix9r> hess_non_zeros;
    std::vector<std::pair<integer, integer>> hess_indices;
    for (integer e = 0; e < element_num; ++e) {
        const Matrix3r vertices = position(Eigen::all, elements_.col(e));
        const Vector3r normal = ComputeNormal(vertices);
        const Matrix3Xr normal_gradient = ComputeNormalGradient(vertices);
        const auto normal_hessian = ComputeNormalHessian(vertices);
        for (integer i = 0; i < 3; ++i) {
            const TriangleEdgeInfo& info = triangle_edge_info_[e][i];
            if (info.other_triangle == -1) continue;
            const Matrix3r other_vertices = position(Eigen::all, elements_.col(info.other_triangle));
            const Vector3r other_normal = ComputeNormal(other_vertices);
            const Matrix3Xr other_normal_gradient = ComputeNormalGradient(other_vertices);

            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const Vector3r dangle_dn0 = ComputeAngleGradient(normal, other_normal).first;
            const Vector3r dangle_dn1 = ComputeAngleGradient(normal, other_normal).second;
            const auto d2angle = ComputeAngleHessian(normal, other_normal);
            const Matrix3r dangle0 = (dangle_dn0.transpose() * normal_gradient).reshaped(3, 3);
            const Matrix3r dangle1 = (dangle_dn1.transpose() * other_normal_gradient).reshaped(3, 3);

            Matrix9r d2angle00 = normal_gradient.transpose() * d2angle[0] * normal_gradient;
            Matrix9r d2angle01 = normal_gradient.transpose() * d2angle[1] * other_normal_gradient;
            Matrix9r d2angle10 = other_normal_gradient.transpose() * d2angle[2] * normal_gradient;
            Matrix9r d2angle11 = other_normal_gradient.transpose() * d2angle[3] * other_normal_gradient;
            for (integer j = 0; j < 9; j++) {
                d2angle00.col(j) += Vector9r(dangle_dn0.transpose() * normal_hessian[j]);
                d2angle11.col(j) += Vector9r(dangle_dn1.transpose() * normal_hessian[j]);
            }

            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;
            const real coefficient = 2 * rest_shape_edge_length * rest_shape_edge_length / diamond_area;

            const Matrix9r hess00 = coefficient * (dangle0.reshaped() * dangle0.reshaped().transpose() + d2angle00 * angle);
            const Matrix9r hess01 = coefficient * (dangle0.reshaped() * dangle1.reshaped().transpose() + d2angle01 * angle);
            const Matrix9r hess10 = coefficient * (dangle1.reshaped() * dangle0.reshaped().transpose() + d2angle10 * angle);
            const Matrix9r hess11 = coefficient * (dangle1.reshaped() * dangle1.reshaped().transpose() + d2angle11 * angle);

            Eigen::VectorXi indices0(9);
            Eigen::VectorXi indices1(9);
            for (integer i = 0; i < 3; ++i){
                for (integer j = 0; j < 3; ++j) {
                    hess.block(elements_(i, e) * 3, elements_(j, e) * 3, 3, 3) += hess00.block(i * 3, j * 3, 3, 3);
                    hess.block(elements_(i, e) * 3, elements_(j, info.other_triangle) * 3, 3, 3) += hess01.block(i * 3, j * 3, 3, 3);
                    hess.block(elements_(i, info.other_triangle) * 3, elements_(j, e) * 3, 3, 3) += hess10.block(i * 3, j * 3, 3, 3);
                    hess.block(elements_(i, info.other_triangle) * 3, elements_(j, info.other_triangle) * 3, 3, 3) += hess11.block(i * 3, j * 3, 3, 3);
                    
                }
            }
        }
    }
    std::vector<Eigen::Triplet<real>> hess_nonzeros;
    for (integer i = 0; i < hess_size; ++i) {
        for (integer j = 0; j < hess_size; ++j) {
            if (hess(i, j) != 0) {
                hess_nonzeros.push_back(Eigen::Triplet<real>(i, j, bending_stiffness_ * hess(i, j)));
            }
        }
    }  
    return FromTriplet(hess_size, hess_size, hess_nonzeros);
    /////////////////////////////////////
}

}
}