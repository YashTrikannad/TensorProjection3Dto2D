//  Yash Trikannad
//  (For SubT DARPA)

// Function to Project 3D Map to 2D Projections
// Inputs - Location, Width of Slice, Axis, Map3d
// Output - 2D Map in the required Axis


#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Proj3dto2d.h"
#include <cmath>
#define WIDTH 3;
#define DIRECTION Y;
#define LOCATION 8;

void projection::Proj3dto2d(const Eigen::TensorMap<Eigen::Tensor<int, 3>> &Map3d,
        Eigen::Tensor<int, 2> &Out, ProjAxis axis,const int &location){

    this->_location = location;
    const Eigen::Tensor<int, 3>::Dimensions& dims = Map3d.dimensions();

    int ceil_width = ceil((this->_width)/2.);

    if((this->_location+ceil_width)>_dims[0] || (this->_location-ceil_width)<0) {
        throw std::runtime_error("The Projection Slice of 2d Maps is out of bounds");
    }

    // YZ Plane
    if (axis == X){
        slice.offsets = {this->_location - (this->_width)/2, 0, 0};
        slice.extents = {(this->_width)/2 + ceil_width, this->_dims[1], this->_dims[2]};
    }

        // XZ Plane3
    else if (axis == Y){
        slice.offsets = {0,  this->_location - (this->_width)/2, 0};
        slice.extents = {this->_dims[0], (this->_width)/2 + ceil_width, this->_dims[2]};
    }
        // Z Axis
    else if (axis == Z) {
        slice.offsets = {0, 0, this->_location - (this->_width)/2};
        slice.extents = {this->_dims[0], this->_dims[1], (this->_width)/2 + ceil_width};
    }
    else{
        throw std::runtime_error("Axis is not Correct. Mention the Perpendicular Axis as X, Y or Z");
    }

    Eigen::Tensor<int, 3> lower_slice = Map3d.slice(slice.offsets, slice.extents);

    this->maxInSlice(lower_slice, axis);

    Out = _maps2d._projection2d;
}

void projection::maxInSlice(const Eigen::Tensor<int, 3> &slice, ProjAxis axis) {

    if (axis == X){
        Eigen::array<int, 1> dims({0});
        _maps2d._projection2d = slice.maximum(dims);
    }
    else if (axis == Y){
        Eigen::array<int, 1> dims({1});
        _maps2d._projection2d = slice.maximum(dims);
    }
    else if(axis == Z){
        Eigen::array<int, 1> dims({2});
        _maps2d._projection2d = slice.maximum(dims);
    }
    else{
        throw std::runtime_error("Axis is not Correct. Mention the Perpendicular Axis as X, Y or Z");
    }

    Eigen::array<int, 2> shuffling({1, 0});
    _maps2d._projection2d = _maps2d._projection2d.shuffle(shuffling).eval();
}


int main() {

//////////////////////////////////////////////////////////////////////////////////////
////////////////               TEST MAP DEFINITION                   /////////////////
//////////////////////////////////////////////////////////////////////////////////////
    Eigen::Tensor<int, 3> t_3d(10, 10, 10);
    t_3d = t_3d.constant(1);

    Eigen::array<int, 3> offsets1 = {0, 0, 4};
    Eigen::array<int, 3> offsets2 = {8, 0, 4};
    Eigen::array<int, 3> offsets3 = {0, 0, 4};
    Eigen::array<int, 3> offsets4 = {0, 8, 4};
    Eigen::array<int, 3> extents1 = {2, 10, 2};
    Eigen::array<int, 3> extents2 = {10, 2, 2};

    Eigen::Tensor<int, 3> slice1 = t_3d.slice(offsets1, extents1).setConstant(2);
    Eigen::Tensor<int, 3> slice2 = t_3d.slice(offsets2, extents1).setConstant(2);
    Eigen::Tensor<int, 3> slice3 = t_3d.slice(offsets3, extents2).setConstant(2);
    Eigen::Tensor<int, 3> slice4 = t_3d.slice(offsets4, extents2).setConstant(2);

    // 3D Map pointed by TensorMap
    Eigen::TensorMap<Eigen::Tensor<int , 3>> map_3d(t_3d.data(), 10, 10, 10);
///////////////////////////////////////////////////////////////////////////////////////

    // Input Parameters
    int ProjectWidth = WIDTH;
    ProjAxis ProjectDir = DIRECTION;
    int ProjLoc = LOCATION;

    const Eigen::Tensor<int, 3>::Dimensions& dims = map_3d.dimensions();
    int dimMap[3];
    dimMap[0] = dims[0];
    dimMap[1] = dims[1];
    dimMap[2] = dims[2];

    projection p(dimMap, ProjectWidth, ProjectDir, ProjLoc);

    Eigen::Tensor<int, 2> ProjectionMap2d;

    p.Proj3dto2d(map_3d, ProjectionMap2d, ProjectDir, ProjLoc);

    std::cout << ProjectionMap2d << std::endl;

    return 0;
    }
