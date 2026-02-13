#pragma once

#include "../driver.hpp"
#include "../boundary.hpp"
#include "../accessor.hpp"
#include "../derivatives.hpp"
#include <cstdio>
#include <sycl/sycl.hpp>

#define ONE 1

struct host_data_container{
    float* vpz;
    float* vsv;
    float* epsilon;
    float* delta;
    float* phi;
    float* theta;
    float* pp;
    float* pc;
    float* qp;
    float* qc;
    float* ch1dxx;
    float* ch1dyy;
    float* ch1dzz;
    float* ch1dxy;
    float* ch1dyz;
    float* ch1dxz;
    float* v2px;
    float* v2pz;
    float* v2sz;
    float* v2pn;
};

struct device_data_container{
    float* ch1dxx;
    float* ch1dyy;
    float* ch1dzz;
    float* ch1dxy;
    float* ch1dyz;
    float* ch1dxz;
    float* v2px;
    float* v2pz;
    float* v2sz;
    float* v2pn;
    float* pDx;
    float* pDy;
    float* qDx;
    float* qDy;
    float* pp;
    float* pc;
    float* qp;
    float* qc;
};

class SYCLDriver : public Driver {

    public:
        SYCLDriver(int argc, char** argv);
        void initialize(
            const int sx_, const int sy_, const int sz_,
            const float dx_, const float dy_, const float dz_,
            const float dt_,
            const int nx_, const int ny_, const int nz_,
            const int bord_, const int absorb_) override;
        void insertSource(int index, float value) override;
        void propagate() override;
        void updateHost() override;
        void finalize() override;
        float* getData() override;

    private:
        sycl::queue q;
        int sx, sy, sz;
        int dx, dy, dz, dt;
        int nx, ny, nz;
        int bord, absorb;
        host_data_container   host_data;
        device_data_container device_data;
};
