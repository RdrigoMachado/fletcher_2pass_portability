#include "sycl_driver.hpp"
#include <cstdio>

SYCLDriver::SYCLDriver(int argc, char** argv)
    : q(sycl::gpu_selector_v) // <--- FORÇA GPU (SYCL 2020)
{
    // Debug: Imprime qual dispositivo foi realmente selecionado
    std::cout << "SYCL Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Opcional: Verifica se é mesmo uma GPU
    if (!q.get_device().is_gpu()) {
        std::cerr << "AVISO CRÍTICO: O código não está rodando na GPU!" << std::endl;
        std::exit(1);
    }
}
void SYCLDriver::initialize(
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
    int msize_vol_extra = msize_vol + 2 * (sx * sy);

    host_data.vpz     = sycl::malloc_host<float>(msize_vol, q);    // p wave speed normal to the simetry plane
    host_data.vsv     = sycl::malloc_host<float>(msize_vol, q);    // sv wave speed normal to the simetry plane
    host_data.epsilon = sycl::malloc_host<float>(msize_vol, q);    // Thomsen isotropic parameter
    host_data.delta   = sycl::malloc_host<float>(msize_vol, q);    // Thomsen isotropic parameter
    host_data.phi     = sycl::malloc_host<float>(msize_vol, q);    // isotropy simetry azimuth angle
    host_data.theta   = sycl::malloc_host<float>(msize_vol, q);    // isotropy simetry azimuth angle
    //Derivatives (?)
    host_data.ch1dxx  = sycl::malloc_host<float>(msize_vol, q);
    host_data.ch1dyy  = sycl::malloc_host<float>(msize_vol, q);
    host_data.ch1dzz  = sycl::malloc_host<float>(msize_vol, q);
    host_data.ch1dxy  = sycl::malloc_host<float>(msize_vol, q);
    host_data.ch1dyz  = sycl::malloc_host<float>(msize_vol, q);
    host_data.ch1dxz  = sycl::malloc_host<float>(msize_vol, q);
    // coeficients of H1 and H2 at PDEs
    host_data.v2px    = sycl::malloc_host<float>(msize_vol, q);
    host_data.v2pz    = sycl::malloc_host<float>(msize_vol, q);
    host_data.v2sz    = sycl::malloc_host<float>(msize_vol, q);
    host_data.v2pn    = sycl::malloc_host<float>(msize_vol, q);
    // Allocate memory for pressure fields
    // By default, initialize with zeros
    host_data.pp      = sycl::malloc_host<float>(msize_vol, q);
    host_data.pc      = sycl::malloc_host<float>(msize_vol, q);
    host_data.qp      = sycl::malloc_host<float>(msize_vol, q);
    host_data.qc      = sycl::malloc_host<float>(msize_vol, q);


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
    device_data.ch1dxx = sycl::malloc_device<float>(msize_vol, q);
    device_data.ch1dyy = sycl::malloc_device<float>(msize_vol, q);
    device_data.ch1dzz = sycl::malloc_device<float>(msize_vol, q);
    device_data.ch1dxy = sycl::malloc_device<float>(msize_vol, q);
    device_data.ch1dyz = sycl::malloc_device<float>(msize_vol, q);
    device_data.ch1dxz = sycl::malloc_device<float>(msize_vol, q);
    device_data.v2px   = sycl::malloc_device<float>(msize_vol, q);
    device_data.v2pz   = sycl::malloc_device<float>(msize_vol, q);
    device_data.v2sz   = sycl::malloc_device<float>(msize_vol, q);
    device_data.v2pn   = sycl::malloc_device<float>(msize_vol, q);

    //Copy
    int msize_vol_bytes = msize_vol * sizeof(float);
    q.memcpy(device_data.ch1dxx, host_data.ch1dxx, msize_vol_bytes);
    q.memcpy(device_data.ch1dyy, host_data.ch1dyy, msize_vol_bytes);
    q.memcpy(device_data.ch1dzz, host_data.ch1dzz, msize_vol_bytes);
    q.memcpy(device_data.ch1dxy, host_data.ch1dxy, msize_vol_bytes);
    q.memcpy(device_data.ch1dyz, host_data.ch1dyz, msize_vol_bytes);
    q.memcpy(device_data.ch1dxz, host_data.ch1dxz, msize_vol_bytes);
    q.memcpy(device_data.v2px,   host_data.v2px,   msize_vol_bytes);
    q.memcpy(device_data.v2pz,   host_data.v2pz,   msize_vol_bytes);
    q.memcpy(device_data.v2sz,   host_data.v2sz,   msize_vol_bytes);
    q.memcpy(device_data.v2pn,   host_data.v2pn,   msize_vol_bytes);

    // Wave field arrays with an extra plan
    // By default is initialized with zeros
    device_data.pDx = sycl::malloc_device<float>(msize_vol, q);
    device_data.pDy = sycl::malloc_device<float>(msize_vol, q);
    device_data.qDx = sycl::malloc_device<float>(msize_vol, q);
    device_data.qDy = sycl::malloc_device<float>(msize_vol, q);
    device_data.pp  = sycl::malloc_device<float>(msize_vol, q);
    device_data.pc  = sycl::malloc_device<float>(msize_vol, q);
    device_data.qp  = sycl::malloc_device<float>(msize_vol, q);
    device_data.qc  = sycl::malloc_device<float>(msize_vol, q);

    q.fill(device_data.pDx, 0, msize_vol);
    q.fill(device_data.pDy, 0, msize_vol);
    q.fill(device_data.qDx, 0, msize_vol);
    q.fill(device_data.qDy, 0, msize_vol);
    q.fill(device_data.pp , 0, msize_vol);
    q.fill(device_data.pc , 0, msize_vol);
    q.fill(device_data.qp , 0, msize_vol);
    q.fill(device_data.qc , 0, msize_vol);
    q.wait();

}

void SYCLDriver::insertSource(int index, float value)
{
    auto local_ptr_dev = this->device_data;
    q.submit([&] (sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(ONE), [=](sycl::id<1> i){
            local_ptr_dev.pc[index] += value;
            local_ptr_dev.qc[index] += value;
        });
    });
    q.wait();
}

void compute_derivatives(
    const int bord,
    const int sx, const int sy, const int sz,
    int strideX, int strideY, float dxinv, float dyinv,
    device_data_container local_ptr_dev, sycl::queue q)
{
    size_t range_x = sx - 2 * bord;
    size_t range_y = sy - 2 * bord;
    int offset = bord;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<2>(range_x, range_y), [=](sycl::item<2> item) {
            const int ix = item.get_id(1) + offset;
            const int iy = item.get_id(0) + offset;
            for (int iz = 0; iz < sz; iz++) {
                const int i = ind(ix, iy, iz, sx, sy);
                local_ptr_dev.pDx[i] = Der1(local_ptr_dev.pc, i, strideX, dxinv);
                local_ptr_dev.pDy[i] = Der1(local_ptr_dev.pc, i, strideY, dyinv);
                local_ptr_dev.qDx[i] = Der1(local_ptr_dev.qc, i, strideX, dxinv);
                local_ptr_dev.qDy[i] = Der1(local_ptr_dev.qc, i, strideY, dyinv);
            }
        });
    });
}

void wave_propagate(
    const int bord,     const float dt,
    const int sx,       const int sy,      const int sz,
    const int strideX,  const int strideY, const int strideZ,
    const float dxxinv, const float dyyinv, const float dzzinv,
    const float dyinv,  const float dzinv,
    device_data_container local_ptr_dev, sycl::queue q)
{
    size_t range_x = sx - (2 * bord);
    size_t range_y = sy - (2 * bord);
    int offset = bord;

    q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::range<2>(range_x, range_y), [=](sycl::item<2> item){
            const int ix = item.get_id(1) + offset;
            const int iy = item.get_id(0) + offset;
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
    });
}

void SYCLDriver::propagate()
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
        dxinv, dyinv, this->device_data, q);

    wave_propagate(bord,    dt,
        sx,      sy,      sz,
        strideX, strideY, strideZ,
        dxxinv,  dyyinv,  dzzinv,
        dyinv,   dzinv, this->device_data, q);
    q.wait();

    std::swap(device_data.pp, device_data.pc);
    std::swap(device_data.qp, device_data.qc);

}

void SYCLDriver::updateHost(){
    size_t total_elements = (size_t)sx * (size_t)sy * (size_t)sz;
    size_t msize_vol_bytes = total_elements * sizeof(float);
    q.memcpy(host_data.pc, device_data.pc, msize_vol_bytes).wait();
}

float* SYCLDriver::getData(){
    return host_data.pc;
}

void SYCLDriver::finalize(){
    sycl::free(device_data.ch1dxx, q);
    sycl::free(device_data.ch1dyy, q);
    sycl::free(device_data.ch1dzz, q);
    sycl::free(device_data.ch1dxy, q);
    sycl::free(device_data.ch1dyz, q);
    sycl::free(device_data.ch1dxz, q);
    sycl::free(device_data.v2px,   q);
    sycl::free(device_data.v2pz,   q);
    sycl::free(device_data.v2sz,   q);
    sycl::free(device_data.v2pn,   q);
    sycl::free(device_data.pDx,    q);
    sycl::free(device_data.pDy,    q);
    sycl::free(device_data.qDx,    q);
    sycl::free(device_data.qDy,    q);
    sycl::free(device_data.pp,     q);
    sycl::free(device_data.pc,     q);
    sycl::free(device_data.qp,     q);
    sycl::free(device_data.qc,     q);

    sycl::free(host_data.vpz,     q);
    sycl::free(host_data.vsv,     q);
    sycl::free(host_data.epsilon, q);
    sycl::free(host_data.delta,   q);
    sycl::free(host_data.phi,     q);
    sycl::free(host_data.theta,   q);
    sycl::free(host_data.pp,      q);
    sycl::free(host_data.pc,      q);
    sycl::free(host_data.qp,      q);
    sycl::free(host_data.qc,      q);
    sycl::free(host_data.ch1dxx,  q);
    sycl::free(host_data.ch1dyy,  q);
    sycl::free(host_data.ch1dzz,  q);
    sycl::free(host_data.ch1dxy,  q);
    sycl::free(host_data.ch1dyz,  q);
    sycl::free(host_data.ch1dxz,  q);
    sycl::free(host_data.v2px,    q);
    sycl::free(host_data.v2pz,    q);
    sycl::free(host_data.v2sz,    q);
    sycl::free(host_data.v2pn,    q);
    q.wait();
}

std::unique_ptr<Driver> createDriver(int argc, char** argv) {
    return std::make_unique<SYCLDriver>(argc, argv);
}
