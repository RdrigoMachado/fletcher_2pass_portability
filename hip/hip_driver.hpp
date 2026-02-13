#pragma once

#include "../driver.hpp"
#include "../boundary.hpp"
#include "../accessor.hpp"    // Contém 'ind' e 'accessor'
#include "../derivatives.hpp" // Contém 'Der1', 'Der2'
#include <cstdio>
#include <hip/hip_runtime.h>

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

class HipDriver : public Driver {

    public:
        HipDriver(int argc, char** argv);
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
        int sx, sy, sz;
        float dx, dy, dz, dt;
        int nx, ny, nz;
        int bord, absorb;
        host_data_container    host_data;
        device_data_container device_data;

        // Armazena o offset calculado para poder liberar a memória corretamente no final
        size_t padding_offset_bytes;
};
