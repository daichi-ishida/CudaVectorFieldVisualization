#pragma once

#include <cuda.h>
#include <vector>

struct CUarrayGrid
{
    CUarrayGrid(){};
    ~CUarrayGrid(){};

    void setup(const unsigned int _xRes, const unsigned int _yRes);
    void clear();

    unsigned int xRes;
    unsigned int yRes;

    CUarray cu_array;
    CUsurfObject surfData;
};

// normal device memory
struct CUdeviceVector
{
    CUdeviceVector(){};
    ~CUdeviceVector(){};

    void setup(const unsigned int _xRes, const unsigned int _yRes);
    void clear();

    void memsetZero();
    void memcpyDtoH(std::vector<float>& data);

    float getValue(const unsigned int index);
    float* getPtr();

    unsigned int xRes;
    unsigned int yRes;

    CUdeviceptr cu_ptr;
};


struct CUarrayGridVector
{
    CUarrayGridVector(){};
    ~CUarrayGridVector(){};

    void setup(const unsigned int _xRes, const unsigned int _yRes);
    void clear();

    unsigned int xRes;
    unsigned int yRes;

    CUarrayGrid x;
    CUarrayGrid y;
};


struct CUarrayStaggeredGrid
{
    CUarrayStaggeredGrid(){};
    ~CUarrayStaggeredGrid(){};

    void setup(const unsigned int _xRes, const unsigned int _yRes);
    void clear();

    unsigned int xRes;
    unsigned int yRes;

    CUarrayGrid x;
    CUarrayGrid y;
};

struct CUarrayCharGrid
{
    CUarrayCharGrid(){};
    ~CUarrayCharGrid(){};

    void setup(const unsigned int _xRes, const unsigned int _yRes);
    void clear();

    unsigned int xRes;
    unsigned int yRes;

    CUarray cu_array;
    CUsurfObject surfData;
};

struct CUdctDeviceVector
{
    CUdctDeviceVector(){};
    ~CUdctDeviceVector(){};

    void setup(const unsigned int _xRes, const unsigned int _yRes);
    void clear();

    void memsetZero();

    float getValue(const unsigned int index);
    float* getPtr();

    unsigned int xRes;
    unsigned int yRes;

    unsigned int xBlockRes;
    unsigned int yBlockRes;

    CUdeviceptr cu_size_ptr;
    CUdeviceptr cu_position_ptr;

    CUdeviceptr cu_coeff_ptr;
};