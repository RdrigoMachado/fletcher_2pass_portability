#include "kokkos_driver.hpp"

KokkosDriver::KokkosDriver(int argc, char** argv) : guard_(argc, argv){}

void KokkosDriver::initialize(
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

    host_data.vpz     = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("vpz",    Kokkos::WithoutInitializing), msize_vol);    // p wave speed normal to the simetry plane
    host_data.vsv     = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("vsv",    Kokkos::WithoutInitializing), msize_vol);    // sv wave speed normal to the simetry plane
    host_data.epsilon = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("epsilon",Kokkos::WithoutInitializing), msize_vol);    // Thomsen isotropic parameter
    host_data.delta   = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("delta",  Kokkos::WithoutInitializing), msize_vol);    // Thomsen isotropic parameter
    host_data.phi     = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("phi",    Kokkos::WithoutInitializing), msize_vol);    // isotropy simetry azimuth angle
    host_data.theta   = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("theta",  Kokkos::WithoutInitializing), msize_vol);    // isotropy simetry azimuth angle
    //Derivatives (?)
    host_data.ch1dxx  = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("ch1dxx", Kokkos::WithoutInitializing), msize_vol);
    host_data.ch1dyy  = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("ch1dyy", Kokkos::WithoutInitializing), msize_vol);
    host_data.ch1dzz  = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("ch1dzz", Kokkos::WithoutInitializing), msize_vol);
    host_data.ch1dxy  = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("ch1dxy", Kokkos::WithoutInitializing), msize_vol);
    host_data.ch1dyz  = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("ch1dyz", Kokkos::WithoutInitializing), msize_vol);
    host_data.ch1dxz  = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("ch1dxz", Kokkos::WithoutInitializing), msize_vol);
    // coeficients of H1 and H2 at PDEs
    host_data.v2px    = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("v2px", Kokkos::WithoutInitializing), msize_vol);
    host_data.v2pz    = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("v2pz", Kokkos::WithoutInitializing), msize_vol);
    host_data.v2sz    = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("v2sz", Kokkos::WithoutInitializing), msize_vol);
    host_data.v2pn    = Kokkos::View<float*, HostMemSpace>(Kokkos::view_alloc("v2pn", Kokkos::WithoutInitializing), msize_vol);
    // Allocate memory for pressure fields
    // By default, initialize with zeros
    host_data.pp      = Kokkos::View<float*, HostMemSpace>("pp",       msize_vol);
    host_data.pc      = Kokkos::View<float*, HostMemSpace>("pc",       msize_vol);
    host_data.qp      = Kokkos::View<float*, HostMemSpace>("qp",       msize_vol);
    host_data.qc      = Kokkos::View<float*, HostMemSpace>("qc",       msize_vol);


    if (SIGMA > MAX_SIGMA) {
        printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n",
    		      SIGMA, MAX_SIGMA);
    }
    for (int i=0; i<sx*sy*sz; i++) {
        host_data.vpz(i)      = 3000.0;
        host_data.epsilon(i)  = 0.24;
        host_data.delta(i)    = 0.1;
        host_data.phi(i)      = 1.0;
        host_data.theta(i)    = atanf(1.0);
        if (SIGMA > MAX_SIGMA) {
           	host_data.vsv(i)  = 0.0;
        } else {
            host_data.vsv(i)  = host_data.vpz(i) * sqrtf(fabsf(host_data.epsilon(i) - host_data.delta(i)) / SIGMA);
        }
    }

    RandomVelocityBoundary(sx, sy, sz,
    			 nx, ny, nz,
    			 bord, absorb,
    			 host_data.vpz, host_data.vsv);

    for (int i=0; i<msize_vol; i++) {
        float sinTheta=sin(host_data.theta(i));
        float cosTheta=cos(host_data.theta(i));
        float sin2Theta=sin(2.0*host_data.theta(i));
        float sinPhi=sin(host_data.phi(i));
        float cosPhi=cos(host_data.phi(i));
        float sin2Phi=sin(2.0*host_data.phi(i));
        host_data.ch1dxx(i)=sinTheta*sinTheta * cosPhi*cosPhi;
        host_data.ch1dyy(i)=sinTheta*sinTheta * sinPhi*sinPhi;
        host_data.ch1dzz(i)=cosTheta*cosTheta;
        host_data.ch1dxy(i)=sinTheta*sinTheta * sin2Phi;
        host_data.ch1dyz(i)=sin2Theta         * sinPhi;
        host_data.ch1dxz(i)=sin2Theta         * cosPhi;
    }

    for (int i=0; i<msize_vol; i++){
        host_data.v2sz(i)=host_data.vsv(i)*host_data.vsv(i);
        host_data.v2pz(i)=host_data.vpz(i)*host_data.vpz(i);
        host_data.v2px(i)=host_data.v2pz(i)*(1.0+2.0*host_data.epsilon(i));
        host_data.v2pn(i)=host_data.v2pz(i)*(1.0+2.0*host_data.delta(i));
    }

    //DEVICE
    //
    //
    //
    device_data.ch1dxx = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev ch1dxx", Kokkos::WithoutInitializing), msize_vol);
    device_data.ch1dyy = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev ch1dyy", Kokkos::WithoutInitializing), msize_vol);
    device_data.ch1dzz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev ch1dzz", Kokkos::WithoutInitializing), msize_vol);
    device_data.ch1dxy = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev ch1dxy", Kokkos::WithoutInitializing), msize_vol);
    device_data.ch1dyz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev ch1dyz", Kokkos::WithoutInitializing), msize_vol);
    device_data.ch1dxz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev ch1dxz", Kokkos::WithoutInitializing), msize_vol);
    device_data.v2px   = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev v2px",   Kokkos::WithoutInitializing), msize_vol);
    device_data.v2pz   = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev v2pz",   Kokkos::WithoutInitializing), msize_vol);
    device_data.v2sz   = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev v2sz",   Kokkos::WithoutInitializing), msize_vol);
    device_data.v2pn   = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc("dev v2pn",   Kokkos::WithoutInitializing), msize_vol);

    //Copy
    Kokkos::deep_copy(device_data.ch1dxx, host_data.ch1dxx);
    Kokkos::deep_copy(device_data.ch1dyy, host_data.ch1dyy);
    Kokkos::deep_copy(device_data.ch1dzz, host_data.ch1dzz);
    Kokkos::deep_copy(device_data.ch1dxy, host_data.ch1dxy);
    Kokkos::deep_copy(device_data.ch1dyz, host_data.ch1dyz);
    Kokkos::deep_copy(device_data.ch1dxz, host_data.ch1dxz);
    Kokkos::deep_copy(device_data.v2px,   host_data.v2px);
    Kokkos::deep_copy(device_data.v2pz,   host_data.v2pz);
    Kokkos::deep_copy(device_data.v2sz,   host_data.v2sz);
    Kokkos::deep_copy(device_data.v2pn,   host_data.v2pn);

    // Wave field arrays with an extra plan
    // By default is initialized with zeros
    device_data.pDx = Kokkos::View<float*, DeviceMemSpace>("dev pDx", msize_vol);
    device_data.pDy = Kokkos::View<float*, DeviceMemSpace>("dev pDy", msize_vol);
    device_data.qDx = Kokkos::View<float*, DeviceMemSpace>("dev qDx", msize_vol);
    device_data.qDy = Kokkos::View<float*, DeviceMemSpace>("dev qDy", msize_vol);

    device_data.pp  = Kokkos::View<float*, DeviceMemSpace>("dev pp",  msize_vol);
    device_data.pc  = Kokkos::View<float*, DeviceMemSpace>("dev pc",  msize_vol);
    device_data.qp  = Kokkos::View<float*, DeviceMemSpace>("dev qp",  msize_vol);
    device_data.qc  = Kokkos::View<float*, DeviceMemSpace>("dev qc",  msize_vol);
}

void KokkosDriver::insertSource(int index, float value)
{
    auto local_ptr_dev = this->device_data;
    Kokkos::parallel_for("Insert Source to grid", ONE, KOKKOS_LAMBDA(int i){
        local_ptr_dev.pc(index) += value;
        local_ptr_dev.qc(index) += value;
    });
    Kokkos::fence();
}

void compute_derivatives(
    const int bord,
    const int sx, const int sy, const int sz,
    int strideX, int strideY, float dxinv, float dyinv,
    device_data_container local_ptr_dev)
{
    size_t start_x = bord;
    size_t start_y = bord;
    size_t end_x = sx - bord;
    size_t end_y = sy - bord;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({start_x, start_y},{end_x, end_y});

    Kokkos::parallel_for("Compute derivatives", policy,
        KOKKOS_LAMBDA(int ix, int iy) {
            for (int iz = 0; iz < sz; iz++) {
                const int i = ind(ix, iy, iz, sx, sy);
                local_ptr_dev.pDx(i)= Der1(local_ptr_dev.pc, i, strideX, dxinv);
                local_ptr_dev.pDy(i)= Der1(local_ptr_dev.pc, i, strideY, dyinv);
                local_ptr_dev.qDx(i)= Der1(local_ptr_dev.qc, i, strideX, dxinv);
                local_ptr_dev.qDy(i)= Der1(local_ptr_dev.qc, i, strideY, dyinv);
            }
        }
    );
}

void wave_propagate(
    const int bord,     const float dt,
    const int sx,       const int sy,      const int sz,
    const int strideX,  const int strideY, const int strideZ,
    const float dxxinv, const float dyyinv, const float dzzinv,
    const float dyinv,  const float dzinv,
    device_data_container local_ptr_dev)
{
    size_t start_x = bord;
    size_t start_y = bord;
    size_t end_x = sx - bord;
    size_t end_y = sy - bord;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({start_x, start_y},{end_x, end_y});

    Kokkos::parallel_for("Propagate", policy,
        KOKKOS_LAMBDA(int ix, int iy) {
            for (int iz = bord; iz < sz - bord; iz++) {
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
                const float cpxx=local_ptr_dev.ch1dxx(i)*pxx;
                const float cpyy=local_ptr_dev.ch1dyy(i)*pyy;
                const float cpzz=local_ptr_dev.ch1dzz(i)*pzz;
                const float cpxy=local_ptr_dev.ch1dxy(i)*pxy;
                const float cpxz=local_ptr_dev.ch1dxz(i)*pxz;
                const float cpyz=local_ptr_dev.ch1dyz(i)*pyz;
                const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
                const float h2p=pxx+pyy+pzz-h1p;
                // xy and xz derivatives of q
                const float qxy = Der1(local_ptr_dev.qDx, i, strideY, dyinv);
                const float qxz = Der1(local_ptr_dev.qDx, i, strideZ, dzinv);
                // yz derivative of q
                const float qyz = Der1(local_ptr_dev.qDy, i, strideZ, dzinv);
                // q second order derivatives
                const float qxx= Der2(local_ptr_dev.qc, i, strideX, dxxinv);
                const float qyy= Der2(local_ptr_dev.qc, i, strideY, dyyinv);
                const float qzz= Der2(local_ptr_dev.qc, i, strideZ, dzzinv);
                // H1(q) and H2(q)
                const float cqxx=local_ptr_dev.ch1dxx(i)*qxx;
                const float cqyy=local_ptr_dev.ch1dyy(i)*qyy;
                const float cqzz=local_ptr_dev.ch1dzz(i)*qzz;
                const float cqxy=local_ptr_dev.ch1dxy(i)*qxy;
                const float cqxz=local_ptr_dev.ch1dxz(i)*qxz;
                const float cqyz=local_ptr_dev.ch1dyz(i)*qyz;
                const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
                const float h2q=qxx+qyy+qzz-h1q;
                // p-q derivatives, H1(p-q) and H2(p-q)
                const float h1pmq=h1p-h1q;
                const float h2pmq=h2p-h2q;
                // rhs of p and q equations
                const float rhsp=local_ptr_dev.v2px(i)*h2p + local_ptr_dev.v2pz(i)*h1q + local_ptr_dev.v2sz(i)*h1pmq;
                const float rhsq=local_ptr_dev.v2pn(i)*h2p + local_ptr_dev.v2pz(i)*h1q - local_ptr_dev.v2sz(i)*h2pmq;
                // new p and q
                local_ptr_dev.pp(i)=2.0f*local_ptr_dev.pc(i) - local_ptr_dev.pp(i) + rhsp*dt*dt;
                local_ptr_dev.qp(i)=2.0f*local_ptr_dev.qc(i) - local_ptr_dev.qp(i) + rhsq*dt*dt;
            }
        }
    );
}

void KokkosDriver::propagate()
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
    Kokkos::fence();

    std::swap(device_data.pp, device_data.pc);
    std::swap(device_data.qp, device_data.qc);

}

void KokkosDriver::updateHost(){
    Kokkos::deep_copy(host_data.pc, device_data.pc);
}

float* KokkosDriver::getData(){
    return host_data.pc.data();
}

void KokkosDriver::finalize(){}

std::unique_ptr<Driver> createDriver(int argc, char** argv) {
    return std::make_unique<KokkosDriver>(argc, argv);
}
