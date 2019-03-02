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

#define WIDTH 2;
#define DIRECTION Y;
#define LOCATION 7;

Eigen::Tensor<int, 2> Proj3dto2d(const Eigen::TensorMap<Eigen::Tensor<int , 3>> &map_3d, const int& width,
                                 const ProjAxis& axis, const int& location_along_axis){

    const Eigen::Tensor<int, 3>::Dimensions& dims = map_3d.dimensions();
    Eigen::Tensor<int, 2> projection_2d;

    // Defining dimensions of the projection maps
    if(axis == X) {
        projection_2d = Eigen::Tensor<int ,2>(dims[1], dims[2]);
        if((location_along_axis + width) > dims[0]-1 || (location_along_axis - width)<0) {
            throw std::runtime_error("The Projection Slice of 2d Maps is out of bounds");
        }
    }
    else if(axis == Y){
        projection_2d = Eigen::Tensor<int ,2>(dims[0], dims[2]);
        if((location_along_axis + width) > dims[1]-1 || (location_along_axis - width)<0) {
            throw std::runtime_error("The Projection Slice of 2d Maps is out of bounds");
        }
    }
    else if(axis == Z){
        projection_2d = Eigen::Tensor<int ,2>(dims[0], dims[1]);
        if((location_along_axis + width) > dims[2]-1 || (location_along_axis - width)<0) {
            throw std::runtime_error("The Projection Slice of 2d Maps is out of bounds");
        }
    }
    else{
        throw std::runtime_error("Define a Proper Axis");
    }

    // Defining Slicing Offsets
    Eigen::array<int, 3> offsets;
    Eigen::array<int, 3> extents;

    // YZ Plane
    if (axis == X){
        offsets = {location_along_axis - width, 0, 0};
        extents = {2*width + 1, int(dims[1]), int(dims[2])};
    }
    // XZ Plane3
    else if (axis == Y){
        offsets = {0,  location_along_axis - width, 0};
        extents = {int(dims[0]), 2*width + 1, int(dims[2])};
    }
    // Z Axis
    else if (axis == Z) {
        offsets = {0, 0, location_along_axis - width};
        extents = {int(dims[0]), int(dims[1]), 2*width + 1};
    }

    Eigen::Tensor<int, 3> slice = map_3d.slice(offsets, extents);
    maxInSlice(slice, axis, projection_2d);

    return projection_2d;
}

void maxInSlice(const Eigen::Tensor<int, 3> &slice, const ProjAxis &axis,
                Eigen::Tensor<int, 2> &projection_2d) {

    if (axis == X){
        Eigen::array<int, 1> dims({0});
        projection_2d = slice.maximum(dims);
    }
    else if (axis == Y){
        Eigen::array<int, 1> dims({1});
        projection_2d = slice.maximum(dims);
    }
    else if(axis == Z){
        Eigen::array<int, 1> dims({2});
        projection_2d = slice.maximum(dims);
    }
    else{
        throw std::runtime_error("Axis is not Correct. Mention the Perpendicular Axis as X, Y or Z");
    }

    Eigen::array<int, 2> shuffling({1, 0});
    projection_2d = projection_2d.shuffle(shuffling).eval();
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
///////////////////////////////////////////////////////////////////////////////////////

    // Input Parameters
    int ProjectWidth = WIDTH;
    ProjAxis ProjectDir = DIRECTION;
    int ProjLoc = LOCATION;

    Eigen::Tensor<int, 2> projection_2d = Proj3dto2d(map_3d, ProjectWidth, ProjectDir, ProjLoc);

    std::cout << projection_2d << '\n';

    return 0;
}
