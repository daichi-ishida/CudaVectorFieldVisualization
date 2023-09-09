#pragma once
#include "grid_types.h"
#include <vector>

#include <cuda.h>


class gridData
{
public:
    gridData(){}
    ~gridData(){}

    void configure();
    void clearAll();

    void swapScalarGrid();
    void swapRenderGrid();

    // gas
    CUarrayGrid density0;
    CUarrayGrid density1;

    bool renderGrid;

    float dx;
    unsigned int xRes;
    unsigned int yRes;

    // time step
    float dt;
    float elapsed_time = 0.0f;


private:
    // memcpy Host to CUarray
    void memcpyHostToCUarray(const std::vector<float>& vector, CUarray& cu_array, const unsigned int _width, const unsigned int _height = 1, const unsigned int _depth = 1);
    
    // memcpy CUarray to Host
    void memcpyCUarrayToHost(const CUarray& cu_array, std::vector<float>& vector, const unsigned int _width, const unsigned int _height = 1, const unsigned int _depth = 1);

    // swap
    void swapGrid(CUarrayGrid& grid1, CUarrayGrid& grid2);
};