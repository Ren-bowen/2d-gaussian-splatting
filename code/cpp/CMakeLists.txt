cmake_minimum_required(VERSION 3.16)

project(backend)

# Project-wide setup.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Configure include and lib paths.
include_directories("${ROOT_DIR}/")
include_directories("${ROOT_DIR}/cpp/")
include_directories("${ROOT_DIR}/external/eigen/")

# Compile the python binding library.
include_directories("${ROOT_DIR}/external/pybind11/include/")
add_subdirectory("${ROOT_DIR}/external/pybind11/" "${ROOT_DIR}/build/pybind11")

file(GLOB_RECURSE binding_cpp_files "./*.cpp")
pybind11_add_module(backend ${binding_cpp_files})