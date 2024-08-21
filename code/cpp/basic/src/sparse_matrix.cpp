#include "basic/include/sparse_matrix.hpp"
#include "basic/include/log.hpp"

namespace backend {

const SparseMatrixXr FromTriplet(const integer row_num, const integer col_num,
    const std::vector<Eigen::Triplet<real>>& nonzeros) {
    SparseMatrixXr mat(row_num, col_num);
    mat.setFromTriplets(nonzeros.begin(), nonzeros.end());
    mat.makeCompressed();
    return mat;
}

const std::vector<Eigen::Triplet<real>> ToTriplet(const SparseMatrixXr& mat) {
    SparseMatrixXr mat_compressed = mat;
    mat_compressed.makeCompressed();
    std::vector<Eigen::Triplet<real>> nonzeros;
    for (integer k = 0; k < static_cast<integer>(mat_compressed.outerSize()); ++k)
        for (SparseMatrixXr::InnerIterator it(mat_compressed, k); it; ++it)
            nonzeros.push_back(Eigen::Triplet<real>(it.row(), it.col(), it.value()));
    return nonzeros;
}

}