#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

enum ProjAxis{X, Y, Z};

Eigen::Tensor<int, 2> Proj3dto2d(const Eigen::TensorMap<Eigen::Tensor<int , 3>> &, const int&, const ProjAxis&,
                                const int&);
void maxInSlice(const Eigen::Tensor<int, 3> &, const ProjAxis &, Eigen::Tensor<int, 2> &);





