cmake_minimum_required(VERSION 3.11)
project(test)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(simulator_bindings PRIVATE ${CMAKE_SOURCE_DIR}/simulators/1_mass_spring/include)

set (Python_ROOT_DIR "/home/renbowen/anaconda3/envs/surfel_splatting")
include_directories(SYSTEM /home/renbowen/anaconda3/lib/python3.11/site-packages/pybind11/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/simulators/1_mass_spring/include)
# find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 REQUIRED)

add_library(simulator SHARED ${CMAKE_SOURCE_DIR}/simulators/1_mass_spring/src/simulator.cu)
target_link_libraries(simulator PRIVATE muda cusolver cublas cusparse sfml-graphics )
pybind11_add_module(handsome test.cpp)
pybind11_add_module(simulators simulator_bindings.cpp)
target_link_libraries(simulators PRIVATE 1_mass_spring)
# target_link_libraries(simulators PRIVATE mass_spring Python3::Python pybind11::module)
