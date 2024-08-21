#include "basic/include/log.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {

void Assert(const bool condition, const std::string& location, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("[" + location + "]" + message);
    }
}

}