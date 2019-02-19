//  Header File for Projection from 3D to 2D
//  Yash Trikannad
//  (For SubT DARPA)

#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

struct slice_coords{
    Eigen::array<int, 3> offsets;
    Eigen::array<int, 3> extents;
};

struct maps2d{
    Eigen::Tensor<int, 2> _projection2d;
    maps2d(){}
    maps2d(int dim1, int dim2)
            :  _projection2d(Eigen::Tensor<int ,2>(dim1, dim2))
    {}
};

enum ProjAxis{X, Y, Z};

class projection: public slice_coords, maps2d{

private:
    maps2d _maps2d;
    int _dims[3] = {0,0,0};
    slice_coords slice;

public:
    int _location;
    int _width;
    ProjAxis _axis;

    projection(){}

    projection(const int dims[],const int width,ProjAxis axis,const int location)
            : _maps2d(dims[0], dims[1]),_width(width), _axis(axis),_location(location)
    {
        this->_dims[0] = dims[0];
        this->_dims[1] = dims[1];
        this->_dims[2] = dims[2];
    }

    void maxInSlice(const Eigen::Tensor<int, 3> &ProjectSlice, ProjAxis axis);

    void Proj3dto2d(const Eigen::TensorMap<Eigen::Tensor<int, 3>> &Map3d,\
     Eigen::Tensor<int, 2> &Out, ProjAxis axis, const int &location);

};








