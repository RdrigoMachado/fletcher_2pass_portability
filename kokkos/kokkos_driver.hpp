#pragma once

#include "../driver.hpp"
#include "../accessor.hpp"
#include "../boundary.hpp"
#include "../derivatives.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <cstdio>

#define ONE 1
using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace   = Kokkos::HostSpace;

struct host_data_container{
    Kokkos::View<float*, HostMemSpace> vpz;
    Kokkos::View<float*, HostMemSpace> vsv;
    Kokkos::View<float*, HostMemSpace> epsilon;
    Kokkos::View<float*, HostMemSpace> delta;
    Kokkos::View<float*, HostMemSpace> phi;
    Kokkos::View<float*, HostMemSpace> theta;
    Kokkos::View<float*, HostMemSpace> pp;
    Kokkos::View<float*, HostMemSpace> pc;
    Kokkos::View<float*, HostMemSpace> qp;
    Kokkos::View<float*, HostMemSpace> qc;
    Kokkos::View<float*, HostMemSpace> ch1dxx;
    Kokkos::View<float*, HostMemSpace> ch1dyy;
    Kokkos::View<float*, HostMemSpace> ch1dzz;
    Kokkos::View<float*, HostMemSpace> ch1dxy;
    Kokkos::View<float*, HostMemSpace> ch1dyz;
    Kokkos::View<float*, HostMemSpace> ch1dxz;
    Kokkos::View<float*, HostMemSpace> v2px;
    Kokkos::View<float*, HostMemSpace> v2pz;
    Kokkos::View<float*, HostMemSpace> v2sz;
    Kokkos::View<float*, HostMemSpace> v2pn;
};

struct device_data_container{
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> ch1dxx;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> ch1dyy;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> ch1dzz;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> ch1dxy;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> ch1dyz;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> ch1dxz;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> v2px;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> v2pz;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> v2sz;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> v2pn;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> pDx;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> pDy;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> qDx;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> qDy;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> pp;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> pc;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> qp;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> qc;
};

class KokkosDriver : public Driver {

    public:
        KokkosDriver(int argc, char** argv);
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
        Kokkos::ScopeGuard guard_;
        int sx, sy, sz;
        int dx, dy, dz, dt;
        int nx, ny, nz;
        int bord, absorb;
        host_data_container   host_data;
        device_data_container device_data;
};
