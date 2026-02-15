#include "raja_driver.hpp"

RajaDriver::RajaDriver(int argc, char** argv) : rm(umpire::ResourceManager::getInstance()) {
    host_allocator    = rm.getAllocator("HOST");
    device_allocator  = rm.getAllocator("DEVICE");

}

void RajaDriver::initialize(
    const int sx_, const int sy_, const int sz_,
    const float dx_, const float dy_, const float dz_,
    const float dt_,
    const int nx_, const int ny_, const int nz_,
    const int bord_, const int absorb_)
{
    sx = sx_;     sy = sy_;    sz = sz_;
    dx = dx_;     dy = dy_;    dz = dz_;
    dt = dt_;
    nx = nx_;     ny = ny_;    nz = nz_;
    bord = bord_; absorb = absorb_;
    int msize_vol = sx * sy * sz;
    int msize_vol_bytes = msize_vol * sizeof(float);

    host_data.vpz     = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));    // p wave speed normal to the simetry plane
    host_data.vsv     = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));    // sv wave speed normal to the simetry plane
    host_data.epsilon = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));    // Thomsen isotropic parameter
    host_data.delta   = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));    // Thomsen isotropic parameter
    host_data.phi     = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));    // isotropy simetry azimuth angle
    host_data.theta   = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));    // isotropy simetry azimuth angle
    //Derivatives (?)
    host_data.ch1dxx  = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.ch1dyy  = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.ch1dzz  = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.ch1dxy  = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.ch1dyz  = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.ch1dxz  = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    // coeficients of H1 and H2 at PDEs
    host_data.v2px    = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.v2pz    = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.v2sz    = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.v2pn    = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    // Allocate memory for pressure fields
    // By default, initialize with zeros
    host_data.pp      = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.pc      = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.qp      = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));
    host_data.qc      = static_cast<float*>(host_allocator.allocate(msize_vol_bytes));


    if (SIGMA > MAX_SIGMA) {
        printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n",
    		      SIGMA, MAX_SIGMA);
    }
    for (int i=0; i<sx*sy*sz; i++) {
        host_data.vpz[i]      = 3000.0;
        host_data.epsilon[i]  = 0.24;
        host_data.delta[i]    = 0.1;
        host_data.phi[i]      = 1.0;
        host_data.theta[i]    = atanf(1.0);
        if (SIGMA > MAX_SIGMA) {
           	host_data.vsv[i]  = 0.0;
        } else {
            host_data.vsv[i]  = host_data.vpz[i] * sqrtf(fabsf(host_data.epsilon[i] - host_data.delta[i]) / SIGMA);
        }
    }

    RandomVelocityBoundary(sx, sy, sz,
    			 nx, ny, nz,
    			 bord, absorb,
    			 host_data.vpz, host_data.vsv);

    for (int i=0; i<msize_vol; i++) {
        float sinTheta=sin(host_data.theta[i]);
        float cosTheta=cos(host_data.theta[i]);
        float sin2Theta=sin(2.0*host_data.theta[i]);
        float sinPhi=sin(host_data.phi[i]);
        float cosPhi=cos(host_data.phi[i]);
        float sin2Phi=sin(2.0*host_data.phi[i]);
        host_data.ch1dxx[i]=sinTheta*sinTheta * cosPhi*cosPhi;
        host_data.ch1dyy[i]=sinTheta*sinTheta * sinPhi*sinPhi;
        host_data.ch1dzz[i]=cosTheta*cosTheta;
        host_data.ch1dxy[i]=sinTheta*sinTheta * sin2Phi;
        host_data.ch1dyz[i]=sin2Theta         * sinPhi;
        host_data.ch1dxz[i]=sin2Theta         * cosPhi;
    }

    for (int i=0; i<msize_vol; i++){
        host_data.v2sz[i]=host_data.vsv[i]*host_data.vsv[i];
        host_data.v2pz[i]=host_data.vpz[i]*host_data.vpz[i];
        host_data.v2px[i]=host_data.v2pz[i]*(1.0+2.0*host_data.epsilon[i]);
        host_data.v2pn[i]=host_data.v2pz[i]*(1.0+2.0*host_data.delta[i]);
    }


    //DEVICE
    //
    //
    //
    device_data.ch1dxx = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.ch1dyy = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.ch1dzz = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.ch1dxy = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.ch1dyz = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.ch1dxz = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.v2px   = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.v2pz   = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.v2sz   = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.v2pn   = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));

    //Copy
    rm.copy(device_data.ch1dxx, host_data.ch1dxx, msize_vol_bytes);
    rm.copy(device_data.ch1dyy, host_data.ch1dyy, msize_vol_bytes);
    rm.copy(device_data.ch1dzz, host_data.ch1dzz, msize_vol_bytes);
    rm.copy(device_data.ch1dxy, host_data.ch1dxy, msize_vol_bytes);
    rm.copy(device_data.ch1dyz, host_data.ch1dyz, msize_vol_bytes);
    rm.copy(device_data.ch1dxz, host_data.ch1dxz, msize_vol_bytes);
    rm.copy(device_data.v2px,   host_data.v2px,   msize_vol_bytes);
    rm.copy(device_data.v2pz,   host_data.v2pz,   msize_vol_bytes);
    rm.copy(device_data.v2sz,   host_data.v2sz,   msize_vol_bytes);
    rm.copy(device_data.v2pn,   host_data.v2pn,   msize_vol_bytes);

    // Wave field arrays with an extra plan
    // By default is initialized with zeros
    device_data.pDx = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.pDy = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.qDx = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.qDy = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.pp  = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.pc  = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.qp  = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));
    device_data.qc  = static_cast<float*>(device_allocator.allocate(msize_vol_bytes));

    rm.memset(device_data.pDx, 0, msize_vol_bytes);
    rm.memset(device_data.pDy, 0, msize_vol_bytes);
    rm.memset(device_data.qDx, 0, msize_vol_bytes);
    rm.memset(device_data.qDy, 0, msize_vol_bytes);
    rm.memset(device_data.pp,  0, msize_vol_bytes);
    rm.memset(device_data.pc,  0, msize_vol_bytes);
    rm.memset(device_data.qp,  0, msize_vol_bytes);
    rm.memset(device_data.qc,  0, msize_vol_bytes);
    GPU_SYNC();

}

void RajaDriver::insertSource(int index, float value)
{
    auto local_ptr_dev = this->device_data;
    RAJA::forall<POLITICA_1D>(RAJA::RangeSegment(0, 1), [=]RAJA_HOST_DEVICE(int i){
        local_ptr_dev.pc[index] += value;
        local_ptr_dev.qc[index] += value;
    });
    GPU_SYNC();
}

void compute_derivatives(
    const int bord,
    const int sx, const int sy, const int sz,
    int strideX, int strideY, float dxinv, float dyinv,
    device_data_container local_ptr_dev)
{
    RAJA::TypedRangeSegment<int> YRange(bord, sy - bord);
    RAJA::TypedRangeSegment<int> XRange(bord, sx - bord);

    RAJA::kernel<POLITICA_XY>(RAJA::make_tuple(XRange, YRange),[=] RAJA_DEVICE (int ix, int iy){
        for (int iz = 0; iz < sz; iz++) {
            const int i = ind(ix, iy, iz, sx, sy);
            local_ptr_dev.pDx[i] = Der1(local_ptr_dev.pc, i, strideX, dxinv);
            local_ptr_dev.pDy[i] = Der1(local_ptr_dev.pc, i, strideY, dyinv);
            local_ptr_dev.qDx[i] = Der1(local_ptr_dev.qc, i, strideX, dxinv);
            local_ptr_dev.qDy[i] = Der1(local_ptr_dev.qc, i, strideY, dyinv);
        }
    });
}

void wave_propagate(
    const int bord,     const float dt,
    const int sx,       const int sy,      const int sz,
    const int strideX,  const int strideY, const int strideZ,
    const float dxxinv, const float dyyinv, const float dzzinv,
    const float dyinv,  const float dzinv,
    device_data_container local_ptr_dev)
{
    int offset = bord;
    RAJA::TypedRangeSegment<int> YRange(offset, sy - bord);
    RAJA::TypedRangeSegment<int> XRange(offset, sx - bord);

    RAJA::kernel<POLITICA_XY>(RAJA::make_tuple(XRange, YRange),[=] RAJA_DEVICE (int ix, int iy){
        for (int iz = offset; iz < sz - offset; iz++) {
            const int i = ind(ix, iy, iz, sx, sy);
            // xy and xz derivatives of p
            const float pxy = Der1(local_ptr_dev.pDx, i, strideY, dyinv);
            const float pxz = Der1(local_ptr_dev.pDx, i, strideZ, dzinv);
            // yz derivative of p
            const float pyz = Der1(local_ptr_dev.pDy, i, strideZ, dzinv);
            // second order derivatives of p
            const float pxx= Der2(local_ptr_dev.pc, i, strideX, dxxinv);
            const float pyy= Der2(local_ptr_dev.pc, i, strideY, dyyinv);
            const float pzz= Der2(local_ptr_dev.pc, i, strideZ, dzzinv);
            // H1(p) and H2(p)
            const float cpxx=local_ptr_dev.ch1dxx[i]*pxx;
            const float cpyy=local_ptr_dev.ch1dyy[i]*pyy;
            const float cpzz=local_ptr_dev.ch1dzz[i]*pzz;
            const float cpxy=local_ptr_dev.ch1dxy[i]*pxy;
            const float cpxz=local_ptr_dev.ch1dxz[i]*pxz;
            const float cpyz=local_ptr_dev.ch1dyz[i]*pyz;
            const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
            const float h2p=pxx+pyy+pzz-h1p;
            // xy and xz derivatives of q
            const float qxy = Der1(local_ptr_dev.qDx, i, strideY, dyinv);
            const float qxz = Der1(local_ptr_dev.qDx, i, strideZ, dzinv);
            // yz derivative of q
            const float qyz = Der1(local_ptr_dev.qDy, i, strideZ, dzinv);
            // q second order derivatives
            const float qzz= Der2(local_ptr_dev.qc, i, strideZ, dzzinv);
            const float qxx= Der2(local_ptr_dev.qc, i, strideX, dxxinv);
            const float qyy= Der2(local_ptr_dev.qc, i, strideY, dyyinv);
            // H1(q) and H2(q)
            const float cqxx=local_ptr_dev.ch1dxx[i]*qxx;
            const float cqyy=local_ptr_dev.ch1dyy[i]*qyy;
            const float cqzz=local_ptr_dev.ch1dzz[i]*qzz;
            const float cqxy=local_ptr_dev.ch1dxy[i]*qxy;
            const float cqxz=local_ptr_dev.ch1dxz[i]*qxz;
            const float cqyz=local_ptr_dev.ch1dyz[i]*qyz;
            const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
            const float h2q=qxx+qyy+qzz-h1q;
            // p-q derivatives, H1(p-q) and H2(p-q)
            const float h1pmq=h1p-h1q;
            const float h2pmq=h2p-h2q;
            // rhs of p and q equations
            const float rhsp=local_ptr_dev.v2px[i]*h2p + local_ptr_dev.v2pz[i]*h1q + local_ptr_dev.v2sz[i]*h1pmq;
            const float rhsq=local_ptr_dev.v2pn[i]*h2p + local_ptr_dev.v2pz[i]*h1q - local_ptr_dev.v2sz[i]*h2pmq;
            // new p and q
            local_ptr_dev.pp[i]=2.0f*local_ptr_dev.pc[i] - local_ptr_dev.pp[i] + rhsp*dt*dt;
            local_ptr_dev.qp[i]=2.0f*local_ptr_dev.qc[i] - local_ptr_dev.qp[i] + rhsq*dt*dt;
        }
    });
}

void RajaDriver::propagate()
{

    const float dxxinv=1.0f/(this->dx*this->dx);
    const float dyyinv=1.0f/(this->dy*this->dy);
    const float dzzinv=1.0f/(this->dz*this->dz);
    const float dxinv=1.0f/this->dx;
    const float dyinv=1.0f/this->dy;
    const float dzinv=1.0f/this->dz;

    const int strideX = 1;
    const int strideY = this->sx;
    const int strideZ = this->sx * this->sy;

    compute_derivatives(
        bord,
        sx, sy, sz,
        strideX, strideY,
        dxinv, dyinv, this->device_data);

    wave_propagate(bord,    dt,
        sx,      sy,      sz,
        strideX, strideY, strideZ,
        dxxinv,  dyyinv,  dzzinv,
        dyinv,   dzinv, this->device_data);
    GPU_SYNC();

    std::swap(device_data.pp, device_data.pc);
    std::swap(device_data.qp, device_data.qc);

}

void RajaDriver::updateHost(){
    size_t total_elements = (size_t)sx * (size_t)sy * (size_t)sz;
    size_t msize_vol_bytes = total_elements * sizeof(float);
    rm.copy(host_data.pc, device_data.pc, msize_vol_bytes);
    GPU_SYNC();
}

float* RajaDriver::getData(){
    return host_data.pc;
}

void RajaDriver::finalize(){
        host_allocator.deallocate(host_data.vpz);
        host_allocator.deallocate(host_data.vsv);
        host_allocator.deallocate(host_data.epsilon);
        host_allocator.deallocate(host_data.delta);
        host_allocator.deallocate(host_data.phi);
        host_allocator.deallocate(host_data.theta);
        host_allocator.deallocate(host_data.pp);
        host_allocator.deallocate(host_data.pc);
        host_allocator.deallocate(host_data.qp);
        host_allocator.deallocate(host_data.qc);
        host_allocator.deallocate(host_data.ch1dxx);
        host_allocator.deallocate(host_data.ch1dyy);
        host_allocator.deallocate(host_data.ch1dzz);
        host_allocator.deallocate(host_data.ch1dxy);
        host_allocator.deallocate(host_data.ch1dyz);
        host_allocator.deallocate(host_data.ch1dxz);
        host_allocator.deallocate(host_data.v2px);
        host_allocator.deallocate(host_data.v2pz);
        host_allocator.deallocate(host_data.v2sz);
        host_allocator.deallocate(host_data.v2pn);

        device_allocator.deallocate(device_data.ch1dxx);
        device_allocator.deallocate(device_data.ch1dyy);
        device_allocator.deallocate(device_data.ch1dzz);
        device_allocator.deallocate(device_data.ch1dxy);
        device_allocator.deallocate(device_data.ch1dyz);
        device_allocator.deallocate(device_data.ch1dxz);
        device_allocator.deallocate(device_data.v2px);
        device_allocator.deallocate(device_data.v2pz);
        device_allocator.deallocate(device_data.v2sz);
        device_allocator.deallocate(device_data.v2pn);
        device_allocator.deallocate(device_data.pDx);
        device_allocator.deallocate(device_data.pDy);
        device_allocator.deallocate(device_data.qDx);
        device_allocator.deallocate(device_data.qDy);
        device_allocator.deallocate(device_data.pp);
        device_allocator.deallocate(device_data.pc);
        device_allocator.deallocate(device_data.qp);
        device_allocator.deallocate(device_data.qc);
        GPU_SYNC();

}

std::unique_ptr<Driver> createDriver(int argc, char** argv) {
    return std::make_unique<RajaDriver>(argc, argv);
}
