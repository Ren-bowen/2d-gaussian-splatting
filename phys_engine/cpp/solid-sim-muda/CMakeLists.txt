cmake_minimum_required(VERSION 3.20)

project(Solid_Sim_Muda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set (Python_ROOT_DIR "/home/renbowen/anaconda3/envs/surfel_splatting")
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
set(VTK_PYTHON_OPTIONAL_LINK OFF)
# set MUDA_BUILD_EXAMPLE to OFF for the subdirectory
set(MUDA_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(muda)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# include(FetchContent)
# FetchContent_Declare(
#     SFML
#     GIT_REPOSITORY https://github.com/SFML/SFML.git
#     GIT_TAG 2.6.x
# )

# set(SFML_CMAKE_ARGS
#     -DCMAKE_POSITION_INDEPENDENT_CODE=ON
#     -DCMAKE_CXX_FLAGS=-fPIC
# )

# FetchContent_MakeAvailable(SFML)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found (include: ${Eigen3_INCLUDE_DIR}, version: ${Eigen3_VERSION_STRING})")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

include_directories(${Eigen3_INCLUDE_DIR})

add_subdirectory(simulators)
