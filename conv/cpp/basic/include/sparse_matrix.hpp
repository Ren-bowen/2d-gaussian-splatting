#ifndef BACKEND_BASIC_SPARSE_MATRIX
#define BACKEND_BASIC_SPARSE_MATRIX

#include "basic/include/config.hpp"

namespace backend {

const SparseMatrixXr FromTriplet(const integer row_num, const integer col_num,
    const std::vector<Eigen::Triplet<real>>& nonzeros);
const std::vector<Eigen::Triplet<real>> ToTriplet(const SparseMatrixXr& mat);

}

#endif