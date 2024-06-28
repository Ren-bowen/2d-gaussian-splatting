cmake_minimum_required(VERSION 3.11) #这个也可以不写

# 查找和添加源文件
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
file(GLOB_RECURSE 1_mass_spring_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")

file(GLOB_RECURSE 1_mass_spring_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
add_library(1_mass_spring SHARED ${1_mass_spring_CU_SOURCE} ${1_mass_spring_CPP_SOURCE})
target_compile_options(1_mass_spring PRIVATE -g)
set_target_properties(1_mass_spring PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(1_mass_spring PRIVATE muda cusolver cublas cusparse )
# 链接库
target_link_libraries(1_mass_spring PRIVATE muda cusolver cublas cusparse sfml-graphics )

# Add a test project.
add_executable(test_1_mass_spring "src/main.cpp")
target_link_libraries(test_1_mass_spring 1_mass_spring)