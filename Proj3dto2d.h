//
// Created by yash on 2/16/19.
//

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

struct slice_coords{

    Eigen::array<int, 3> offsets_lower;
    Eigen::array<int, 3> extents_lower;
    Eigen::array<int, 3> offsets_upper;
    Eigen::array<int, 3> extents_upper;

};

struct maps2d{

    Eigen::Tensor<int, 2> _projection2d_lower;
    Eigen::Tensor<int, 2> _projection2d_upper;

    maps2d(){}

    maps2d(int dim1, int dim2)
    :  _projection2d_lower(Eigen::Tensor<int ,2>(dim1, dim2)), _projection2d_upper(Eigen::Tensor<int ,2>(dim1, dim2))
    {}
};

