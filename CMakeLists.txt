cmake_minimum_required(VERSION 3.13)
project(TensorProjection)

set(CMAKE_CXX_STANDARD 14)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(TensorProjection Proj3dto2d.cpp Proj3dto2d.h)
target_link_libraries (TensorProjection Eigen3::Eigen)
