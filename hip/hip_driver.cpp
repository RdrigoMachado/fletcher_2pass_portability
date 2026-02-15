#include "hip_driver.hpp"
#include <cstdio>
#include <algorithm>

HipDriver::HipDriver(int argc, char** argv) : padding_offset_bytes(0) {}

void HipDriver::initialize(
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

    // --- HOST ALLOCATION ---
    hipHostMalloc(&host_data.vpz,      msize_vol_bytes);
    hipHostMalloc(&host_data.vsv,      msize_vol_bytes);
    hipHostMalloc(&host_data.epsilon,  msize_vol_bytes);
    hipHostMalloc(&host_data.delta,    msize_vol_bytes);
    hipHostMalloc(&host_data.phi,      msize_vol_bytes);
    hipHostMalloc(&host_data.theta,    msize_vol_bytes);

    hipHostMalloc(&host_data.ch1dxx, msize_vol_bytes);
    hipHostMalloc(&host_data.ch1dyy, msize_vol_bytes);
    hipHostMalloc(&host_data.ch1dzz, msize_vol_bytes);
    hipHostMalloc(&host_data.ch1dxy, msize_vol_bytes);
    hipHostMalloc(&host_data.ch1dyz, msize_vol_bytes);
    hipHostMalloc(&host_data.ch1dxz, msize_vol_bytes);

    hipHostMalloc(&host_data.v2px, msize_vol_bytes);
    hipHostMalloc(&host_data.v2pz, msize_vol_bytes);
    hipHostMalloc(&host_data.v2sz, msize_vol_bytes);
    hipHostMalloc(&host_data.v2pn, msize_vol_bytes);

    hipHostMalloc(&host_data.pp, msize_vol_bytes);
    hipHostMalloc(&host_data.pc, msize_vol_bytes);
    hipHostMalloc(&host_data.qp, msize_vol_bytes);
    hipHostMalloc(&host_data.qc, msize_vol_bytes);

    if (SIGMA > MAX_SIGMA) {
        printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n",
                  SIGMA, MAX_SIGMA);
    }

    for (int i=0; i<sx*sy*sz; i++) {
        host_data.vpz[i]      = 3000.0f;
        host_data.epsilon[i]  = 0.24f;
        host_data.delta[i]    = 0.1f;
        host_data.phi[i]      = 1.0f;
        host_data.theta[i]    = atanf(1.0f);
        if (SIGMA > MAX_SIGMA) {
               host_data.vsv[i]  = 0.0f;
        } else {
            host_data.vsv[i]  = host_data.vpz[i] * sqrtf(fabsf(host_data.epsilon[i] - host_data.delta[i]) / SIGMA);
        }
    }

    RandomVelocityBoundary(sx, sy, sz,
                nx, ny, nz,
                bord, absorb,
                host_data.vpz, host_data.vsv);

    for (int i=0; i<msize_vol; i++) {
        float sinTheta=sinf(host_data.theta[i]);
        float cosTheta=cosf(host_data.theta[i]);
        float sin2Theta=sinf(2.0f*host_data.theta[i]);
        float sinPhi=sinf(host_data.phi[i]);
        float cosPhi=cosf(host_data.phi[i]);
        float sin2Phi=sinf(2.0f*host_data.phi[i]);
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
        host_data.v2px[i]=host_data.v2pz[i]*(1.0f+2.0f*host_data.epsilon[i]);
        host_data.v2pn[i]=host_data.v2pz[i]*(1.0f+2.0f*host_data.delta[i]);
    }

    // --- DEVICE ALLOCATION (COEFFICIENTS - NO PADDING) ---
    hipMalloc(&device_data.ch1dxx, msize_vol_bytes);
    hipMalloc(&device_data.ch1dyy, msize_vol_bytes);
    hipMalloc(&device_data.ch1dzz, msize_vol_bytes);
    hipMalloc(&device_data.ch1dxy, msize_vol_bytes);
    hipMalloc(&device_data.ch1dyz, msize_vol_bytes);
    hipMalloc(&device_data.ch1dxz, msize_vol_bytes);
    hipMalloc(&device_data.v2px,   msize_vol_bytes);
    hipMalloc(&device_data.v2pz,   msize_vol_bytes);
    hipMalloc(&device_data.v2sz,   msize_vol_bytes);
    hipMalloc(&device_data.v2pn,   msize_vol_bytes);

    hipMemcpy(device_data.ch1dxx, host_data.ch1dxx, msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.ch1dyy, host_data.ch1dyy, msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.ch1dzz, host_data.ch1dzz, msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.ch1dxy, host_data.ch1dxy, msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.ch1dyz, host_data.ch1dyz, msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.ch1dxz, host_data.ch1dxz, msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.v2px,   host_data.v2px,   msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.v2pz,   host_data.v2pz,   msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.v2sz,   host_data.v2sz,   msize_vol_bytes, hipMemcpyHostToDevice);
    hipMemcpy(device_data.v2pn,   host_data.v2pn,   msize_vol_bytes, hipMemcpyHostToDevice);

    // --- DEVICE ALLOCATION (WAVE FIELDS - WITH PADDING) ---
    // Stencil raio 4 exige padding de pelo menos 4 indices
    int padding_z = 4;
    // O offset total em bytes: padding * (elementos por plano XY) * sizeof(float)
    padding_offset_bytes = (size_t)padding_z * (size_t)sx * (size_t)sy * sizeof(float);
    size_t total_alloc_bytes = msize_vol_bytes + 2 * padding_offset_bytes;

    float *raw_pDx, *raw_pDy, *raw_qDx, *raw_qDy, *raw_pp, *raw_pc, *raw_qp, *raw_qc;

    hipMalloc(&raw_pDx, total_alloc_bytes);
    hipMalloc(&raw_pDy, total_alloc_bytes);
    hipMalloc(&raw_qDx, total_alloc_bytes);
    hipMalloc(&raw_qDy, total_alloc_bytes);
    hipMalloc(&raw_pp,  total_alloc_bytes);
    hipMalloc(&raw_pc,  total_alloc_bytes);
    hipMalloc(&raw_qp,  total_alloc_bytes);
    hipMalloc(&raw_qc,  total_alloc_bytes);

    hipMemset(raw_pDx, 0, total_alloc_bytes);
    hipMemset(raw_pDy, 0, total_alloc_bytes);
    hipMemset(raw_qDx, 0, total_alloc_bytes);
    hipMemset(raw_qDy, 0, total_alloc_bytes);
    hipMemset(raw_pp,  0, total_alloc_bytes);
    hipMemset(raw_pc,  0, total_alloc_bytes);
    hipMemset(raw_qp,  0, total_alloc_bytes);
    hipMemset(raw_qc,  0, total_alloc_bytes);

    // Avança o ponteiro para a area segura
    // Aritmetica de ponteiro char* para somar bytes, depois cast para float*
    device_data.pDx = (float*)((char*)raw_pDx + padding_offset_bytes);
    device_data.pDy = (float*)((char*)raw_pDy + padding_offset_bytes);
    device_data.qDx = (float*)((char*)raw_qDx + padding_offset_bytes);
    device_data.qDy = (float*)((char*)raw_qDy + padding_offset_bytes);
    device_data.pp  = (float*)((char*)raw_pp  + padding_offset_bytes);
    device_data.pc  = (float*)((char*)raw_pc  + padding_offset_bytes);
    device_data.qp  = (float*)((char*)raw_qp  + padding_offset_bytes);
    device_data.qc  = (float*)((char*)raw_qc  + padding_offset_bytes);

    hipDeviceSynchronize();
}

__global__ void kernel_InsertSource(const float value, const int index,
                                float * __restrict__ qp, float * __restrict__ qc)
{
    qp[index]+=value;
    qc[index]+=value;
}

void HipDriver::insertSource(int index, float value)
{
    kernel_InsertSource<<<1, 1>>> (value, index, device_data.pc, device_data.qc);
    hipDeviceSynchronize();
}

__global__ void kernel_compute_derivatives(
    const int bord,
    const int sx, const int sy, const int sz,
    const int range_x, const int range_y,
    const int strideX, const int strideY, const float dxinv, const float dyinv,
    float* pDx, float* qDx,
    float* pDy, float* qDy,
    const float* pp, const float* pc,
    const float* qp, const float* qc)
{
  const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_x >= range_x || idx_y >= range_y) return;

  const int ix = idx_x + bord;
  const int iy = idx_y + bord;

  for (int iz=0; iz<sz; iz++) {
      const int i = ind(ix,iy,iz, sx, sy);
      pDx[i]= Der1(pc, i, strideX, dxinv);
      pDy[i]= Der1(pc, i, strideY, dyinv);
      qDx[i]= Der1(qc, i, strideX, dxinv);
      qDy[i]= Der1(qc, i, strideY, dyinv);
  }
}

__global__ void kernel_wave_propagate(
    const int bord,      const float dt,
    const int sx,        const int sy,       const int sz,
    const int range_x,   const int range_y,
    const int strideX,   const int strideY, const int strideZ,
    const float dxxinv,  const float dyyinv, const float dzzinv,
    const float dyinv,   const float dzinv,
    const float* ch1dxx, const float* ch1dyy,
    const float* ch1dzz, const float* ch1dxy,
    const float* ch1dyz, const float* ch1dxz,
    const float* pDx,    const float* qDx,
    const float* pDy,    const float* qDy,
    const float* v2px,   const float* v2pz,
    const float* v2sz,   const float* v2pn,
    float* pp,           const float* pc,
    float* qp,           const float* qc)
{
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= range_x || idx_y >= range_y) return;

    const int ix = idx_x + bord;
    const int iy = idx_y + bord;

    for (int iz=bord; iz<sz-bord; iz++) {
        const int i = ind(ix,iy,iz, sx, sy);

        const float pxy = Der1(pDx, i, strideY, dyinv);
        const float pxz = Der1(pDx, i, strideZ, dzinv);
        const float pyz = Der1(pDy, i, strideZ, dzinv);

        const float pxx= Der2(pc, i, strideX, dxxinv);
        const float pyy= Der2(pc, i, strideY, dyyinv);
        const float pzz= Der2(pc, i, strideZ, dzzinv);

        const float cpxx=ch1dxx[i]*pxx;
        const float cpyy=ch1dyy[i]*pyy;
        const float cpzz=ch1dzz[i]*pzz;
        const float cpxy=ch1dxy[i]*pxy;
        const float cpxz=ch1dxz[i]*pxz;
        const float cpyz=ch1dyz[i]*pyz;
        const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
        const float h2p=pxx+pyy+pzz-h1p;

        const float qxy = Der1(qDx, i, strideY, dyinv);
        const float qxz = Der1(qDx, i, strideZ, dzinv);
        const float qyz = Der1(qDy, i, strideZ, dzinv);

        const float qxx= Der2(qc, i, strideX, dxxinv);
        const float qyy= Der2(qc, i, strideY, dyyinv);
        const float qzz= Der2(qc, i, strideZ, dzzinv);

        const float cqxx=ch1dxx[i]*qxx;
        const float cqyy=ch1dyy[i]*qyy;
        const float cqzz=ch1dzz[i]*qzz;
        const float cqxy=ch1dxy[i]*qxy;
        const float cqxz=ch1dxz[i]*qxz;
        const float cqyz=ch1dyz[i]*qyz;
        const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
        const float h2q=qxx+qyy+qzz-h1q;

        const float h1pmq=h1p-h1q;
        const float h2pmq=h2p-h2q;

        const float rhsp=v2px[i]*h2p + v2pz[i]*h1q + v2sz[i]*h1pmq;
        const float rhsq=v2pn[i]*h2p + v2pz[i]*h1q - v2sz[i]*h2pmq;

        pp[i]=2.0f*pc[i] - pp[i] + rhsp*dt*dt;
        qp[i]=2.0f*qc[i] - qp[i] + rhsq*dt*dt;
    }
}

void HipDriver::propagate()
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

    int range_x = sx - 2 * bord;
    int range_y = sy - 2 * bord;

    dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
    dim3 numBlocks((range_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (range_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_compute_derivatives<<<numBlocks, threadsPerBlock>>>(
        bord,
        sx, sy, sz,
        range_x, range_y,
        strideX, strideY,
        dxinv, dyinv,
        device_data.pDx, device_data.qDx,
        device_data.pDy, device_data.qDy,
        device_data.pp,  device_data.pc,
        device_data.qp,  device_data.qc);

    hipDeviceSynchronize();

    kernel_wave_propagate<<<numBlocks, threadsPerBlock>>>(
        bord,    dt,
        sx,      sy,      sz,
        range_x, range_y,
        strideX, strideY, strideZ,
        dxxinv,  dyyinv,  dzzinv,
        dyinv,   dzinv,
        device_data.ch1dxx, device_data.ch1dyy,
        device_data.ch1dzz, device_data.ch1dxy,
        device_data.ch1dyz, device_data.ch1dxz,
        device_data.pDx,    device_data.qDx,
        device_data.pDy,    device_data.qDy,
        device_data.v2px,   device_data.v2pz,
        device_data.v2sz,   device_data.v2pn,
        device_data.pp, device_data.pc,
        device_data.qp, device_data.qc);

    hipDeviceSynchronize();

    std::swap(device_data.pp, device_data.pc);
    std::swap(device_data.qp, device_data.qc);
}

void HipDriver::updateHost(){
    size_t total_elements = (size_t)sx * sy * sz;
    size_t msize_vol_bytes = total_elements * sizeof(float);
    hipMemcpy(host_data.pc, device_data.pc, msize_vol_bytes, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
}

float* HipDriver::getData(){
    return host_data.pc;
}

void HipDriver::finalize(){
    hipFree(device_data.ch1dxx);
    hipFree(device_data.ch1dyy);
    hipFree(device_data.ch1dzz);
    hipFree(device_data.ch1dxy);
    hipFree(device_data.ch1dyz);
    hipFree(device_data.ch1dxz);
    hipFree(device_data.v2px);
    hipFree(device_data.v2pz);
    hipFree(device_data.v2sz);
    hipFree(device_data.v2pn);

    // Para liberar memoria com padding, precisamos voltar o ponteiro para o inicio original
    // Usamos aritmetica char* para garantir precisão em bytes
    hipFree((float*)((char*)device_data.pDx - padding_offset_bytes));
    hipFree((float*)((char*)device_data.pDy - padding_offset_bytes));
    hipFree((float*)((char*)device_data.qDx - padding_offset_bytes));
    hipFree((float*)((char*)device_data.qDy - padding_offset_bytes));
    hipFree((float*)((char*)device_data.pp - padding_offset_bytes));
    hipFree((float*)((char*)device_data.pc - padding_offset_bytes));
    hipFree((float*)((char*)device_data.qp - padding_offset_bytes));
    hipFree((float*)((char*)device_data.qc - padding_offset_bytes));

    hipHostFree(host_data.vpz);
    hipHostFree(host_data.vsv);
    hipHostFree(host_data.epsilon);
    hipHostFree(host_data.delta);
    hipHostFree(host_data.phi);
    hipHostFree(host_data.theta);
    hipHostFree(host_data.pp);
    hipHostFree(host_data.pc);
    hipHostFree(host_data.qp);
    hipHostFree(host_data.qc);
    hipHostFree(host_data.ch1dxx);
    hipHostFree(host_data.ch1dyy);
    hipHostFree(host_data.ch1dzz);
    hipHostFree(host_data.ch1dxy);
    hipHostFree(host_data.ch1dyz);
    hipHostFree(host_data.ch1dxz);
    hipHostFree(host_data.v2px);
    hipHostFree(host_data.v2pz);
    hipHostFree(host_data.v2sz);
    hipHostFree(host_data.v2pn);
    hipDeviceSynchronize();
}

std::unique_ptr<Driver> createDriver(int argc, char** argv) {
    return std::make_unique<HipDriver>(argc, argv);
}
