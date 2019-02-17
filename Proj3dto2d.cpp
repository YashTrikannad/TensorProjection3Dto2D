#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Proj3dto2d.h"

class projection: public slice_coords, maps2d{

private:
    maps2d _maps2d;
    std::string _plane;
    int _location;
    int _width;
    int _dims[3] = {0,0,0};
    slice_coords slice;

public:
    projection(){}

    projection(const int dims[],const int width,const std::string &plane,const int z_location)
            : _maps2d(dims[0], dims[1]),_plane(plane),_location(z_location),_width(width)
    {
        this->_dims[0] = dims[0];
        this->_dims[1] = dims[1];
        this->_dims[2] = dims[2];
    }

    void maxInSlice(const Eigen::Tensor<int, 3> &slice, std::string s) {

        const Eigen::Tensor<int, 3>::Dimensions &dims_slice = slice.dimensions();

        if (this->_plane == "XY") {
            if (s == "l") {

                for (size_t i = 0; i < this->_dims[0]; i++) {
                    for (size_t j = 0; j < this->_dims[1]; j++) {
                        int max = 0;
                        for (size_t k = 0; k < dims_slice[2]; k++) {
                            if (slice(i, j, k) > max) {
                                max = slice(i, j, k);
                            }
                        }
                        _maps2d._projection2d_lower(i, j) = max;
                    }
                }
                return;
            } else if (s == "u") {
                for (size_t i = 0; i < this->_dims[0]; i++) {
                    for (size_t j = 0; j < this->_dims[1]; j++) {
                        int max = 0;
                        for (size_t k = 0; k < dims_slice[2]; k++) {
                            int x = slice(i, j, k);
                            if (slice(i, j, k) > max) {
                                max = slice(i, j, k);
                            }
                        }
                        _maps2d._projection2d_upper(i, j) = max;
                    }
                }
                return;
            }
        }

        if (this->_plane == "XZ") {
            if (s == "l") {

                for (size_t i = 0; i < this->_dims[2]; i++) {
                    for (size_t j = 0; j < this->_dims[0]; j++) {
                        int max = 0;
                        for (size_t k = 0; k < dims_slice[1]; k++) {
                            if (slice(j, k, i) > max) {
                                max = slice(j, k, i);
                            }
                        }
                        _maps2d._projection2d_lower(i, j) = max;
                    }
                }
                return;
            } else if (s == "u") {
                for (size_t i = 0; i < this->_dims[2]; i++) {
                    for (size_t j = 0; j < this->_dims[0]; j++) {
                        int max = 0;
                        for (size_t k = 0; k < dims_slice[1]; k++) {
                            int x = slice(j, k, i);
                            if (slice(j, k, i) > max) {
                                max = slice(j, k, i);
                            }
                        }
                        _maps2d._projection2d_upper(i, j) = max;
                    }
                }
                return;
            }
        }

        if (this->_plane == "YZ") {
            if (s == "l") {

                for (size_t i = 0; i < this->_dims[1]; i++) {
                    for (size_t j = 0; j < this->_dims[2]; j++) {
                        int max = 0;
                        for (size_t k = 0; k < dims_slice[0]; k++) {
                            if (slice(k, i, j) > max) {
                                max = slice(k, i, j);
                            }
                        }
                        _maps2d._projection2d_lower(j, i) = max;
                    }
                }
                return;
            } else if (s == "u") {
                for (size_t i = 0; i < this->_dims[1]; i++) {
                    for (size_t j = 0; j < this->_dims[2]; j++) {
                        int max = 0;
                        for (size_t k = 0; k < dims_slice[0]; k++) {
                            if (slice(k, i, j) > max) {
                                max = slice(k, i, j);
                            }
                        }
                        _maps2d._projection2d_upper(j, i) = max;
                    }
                }
                return;
            }
        }
    }

    void Proj3dto2d(const Eigen::TensorMap<Eigen::Tensor<int, 3>> &Map3d,\
     Eigen::Tensor<int, 2> &outLower,Eigen::Tensor<int, 2> &outUpper){

        const Eigen::Tensor<int, 3>::Dimensions& dims = Map3d.dimensions();

        int ceil_width = ceil((this->_width)/2.);

        // Lower Slice does not contain the current location. Should I add it ?
        // Upper Slice is the current location + width/2

        // XY Plane
        if (this->_plane == "XY") {
            slice.offsets_lower = {0, 0, this->_location - (this->_width)/2};
            slice.extents_lower = {this->_dims[0], this->_dims[1], (this->_width)/2};
            slice.offsets_upper = {0, 0, this->_location};
            slice.extents_upper = {this->_dims[0], this->_dims[1], ceil_width};
        }

        // YZ Plane
        else if (this->_plane == "YZ"){
            slice.offsets_lower = {this->_location - (this->_width)/2, 0, 0};
            slice.extents_lower = {(this->_width)/2, this->_dims[1], this->_dims[2]};
            slice.offsets_upper = {this->_location, 0, 0};
            slice.extents_upper = {ceil_width, this->_dims[1], this->_dims[2]};
        }

        // XZ Plane3
        else if (this->_plane == "XZ"){
            slice.offsets_lower = {0,  this->_location - (this->_width)/2, 0};
            slice.extents_lower = {this->_dims[0], (this->_width)/2, this->_dims[2]};
            slice.offsets_upper = {0, this->_location, 0};
            slice.extents_upper = {this->_dims[0],  ceil_width, this->_dims[2]};
        }

        Eigen::Tensor<int, 3> lower_slice = Map3d.slice(slice.offsets_lower, slice.extents_lower);
        Eigen::Tensor<int, 3> upper_slice = Map3d.slice(slice.offsets_upper, slice.extents_upper);

        this->maxInSlice(lower_slice, "l");
        this->maxInSlice(upper_slice, "u");

        outLower = _maps2d._projection2d_lower;
        outUpper = _maps2d._projection2d_upper;

    }

};

int main() {

// Populate the input voxel map
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

    // 2D Map stored as TensorMap
    Eigen::TensorMap<Eigen::Tensor<int , 3>> map_3d(t_3d.data(), 10, 10, 10);

    // Input Parameters
    int width = 5;
    std::string plane = "XY";
    int location = 5;

    const Eigen::Tensor<int, 3>::Dimensions& dims = map_3d.dimensions();
    int dimMap[3];
    dimMap[0] = dims[0];
    dimMap[1] = dims[1];
    dimMap[2] = dims[2];

    projection p(dimMap, width, plane, location);

    Eigen::Tensor<int, 2> projection2d_lower;
    Eigen::Tensor<int, 2> projection2d_upper;

    p.Proj3dto2d(map_3d, projection2d_lower, projection2d_upper);

    std::cout << projection2d_lower << std::endl;
    std::cout << projection2d_upper <<std::endl;

    return 0;
    }
