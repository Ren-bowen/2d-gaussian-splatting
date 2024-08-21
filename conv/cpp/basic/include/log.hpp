#ifndef BACKEND_BASIC_LOG
#define BACKEND_BASIC_LOG

#include "basic/include/config.hpp"

namespace backend {

void Assert(const bool condition, const std::string& location, const std::string& message);

}

#endif