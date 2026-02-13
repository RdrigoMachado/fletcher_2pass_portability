#include "cuda_driver.hpp"
#include <cstdio>


CudaDriver::CudaDriver(int argc, char** argv){}

void CudaDriver::initialize(
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
    cudaMallocHost(&host_data.vpz,      msize_vol_bytes);
    cudaMallocHost(&host_data.vsv,      msize_vol_bytes);
    cudaMallocHost(&host_data.epsilon,  msize_vol_bytes);
    cudaMallocHost(&host_data.delta,    msize_vol_bytes);
    cudaMallocHost(&host_data.phi,      msize_vol_bytes);
    cudaMallocHost(&host_data.theta,    msize_vol_bytes);

    cudaMallocHost(&host_data.ch1dxx, msize_vol_bytes);
    cudaMallocHost(&host_data.ch1dyy, msize_vol_bytes);
    cudaMallocHost(&host_data.ch1dzz, msize_vol_bytes);
    cudaMallocHost(&host_data.ch1dxy, msize_vol_bytes);
    cudaMallocHost(&host_data.ch1dyz, msize_vol_bytes);
    cudaMallocHost(&host_data.ch1dxz, msize_vol_bytes);

    cudaMallocHost(&host_data.v2px, msize_vol_bytes);
    cudaMallocHost(&host_data.v2pz, msize_vol_bytes);
    cudaMallocHost(&host_data.v2sz, msize_vol_bytes);
    cudaMallocHost(&host_data.v2pn, msize_vol_bytes);

    cudaMallocHost(&host_data.pp, msize_vol_bytes);
    cudaMallocHost(&host_data.pc, msize_vol_bytes);
    cudaMallocHost(&host_data.qp, msize_vol_bytes);
    cudaMallocHost(&host_data.qc, msize_vol_bytes);

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

    // --- DEVICE ALLOCATION ---
    cudaMalloc(&device_data.ch1dxx, msize_vol_bytes);
    cudaMalloc(&device_data.ch1dyy, msize_vol_bytes);
    cudaMalloc(&device_data.ch1dzz, msize_vol_bytes);
    cudaMalloc(&device_data.ch1dxy, msize_vol_bytes);
    cudaMalloc(&device_data.ch1dyz, msize_vol_bytes);
    cudaMalloc(&device_data.ch1dxz, msize_vol_bytes);
    cudaMalloc(&device_data.v2px,   msize_vol_bytes);
    cudaMalloc(&device_data.v2pz,   msize_vol_bytes);
    cudaMalloc(&device_data.v2sz,   msize_vol_bytes);
    cudaMalloc(&device_data.v2pn,   msize_vol_bytes);

    cudaMemcpy(device_data.ch1dxx, host_data.ch1dxx, msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.ch1dyy, host_data.ch1dyy, msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.ch1dzz, host_data.ch1dzz, msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.ch1dxy, host_data.ch1dxy, msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.ch1dyz, host_data.ch1dyz, msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.ch1dxz, host_data.ch1dxz, msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.v2px,   host_data.v2px,   msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.v2pz,   host_data.v2pz,   msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.v2sz,   host_data.v2sz,   msize_vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.v2pn,   host_data.v2pn,   msize_vol_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&device_data.pDx, msize_vol_bytes);
    cudaMalloc(&device_data.pDy, msize_vol_bytes);
    cudaMalloc(&device_data.qDx, msize_vol_bytes);
    cudaMalloc(&device_data.qDy, msize_vol_bytes);
    cudaMalloc(&device_data.pp,  msize_vol_bytes);
    cudaMalloc(&device_data.pc,  msize_vol_bytes);
    cudaMalloc(&device_data.qp,  msize_vol_bytes);
    cudaMalloc(&device_data.qc,  msize_vol_bytes);

    cudaMemset(device_data.pDx, 0, msize_vol_bytes);
    cudaMemset(device_data.pDy, 0, msize_vol_bytes);
    cudaMemset(device_data.qDx, 0, msize_vol_bytes);
    cudaMemset(device_data.qDy, 0, msize_vol_bytes);
    cudaMemset(device_data.pp,  0, msize_vol_bytes);
    cudaMemset(device_data.pc,  0, msize_vol_bytes);
    cudaMemset(device_data.qp,  0, msize_vol_bytes);
    cudaMemset(device_data.qc,  0, msize_vol_bytes);

    cudaDeviceSynchronize();
}

__global__ void kernel_InsertSource(const float value, const int index,
                                float * __restrict__ qp, float * __restrict__ qc)
{
    qp[index]+=value;
    qc[index]+=value;
}

void CudaDriver::insertSource(int index, float value)
{
    kernel_InsertSource<<<1, 1>>> (value, index, device_data.pc, device_data.qc);
    cudaDeviceSynchronize();
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
  // Mapeia 0..range (sem borda)
  const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Verifica se estamos dentro da area "ativa" (sem as bordas)
  if (idx_x >= range_x || idx_y >= range_y) return;

  // Aplica o offset para encontrar o indice real na memoria
  const int ix = idx_x + bord;
  const int iy = idx_y + bord;

  // Derivadas geralmente varrem todo o Z (0 a sz) ou tambem pulam borda?
  // No seu SYCL estava de 0 a sz. Mantive assim.
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

    // Offset para pular 'bord'
    const int ix = idx_x + bord;
    const int iy = idx_y + bord;

    // Pula borda em Z tambem
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

void CudaDriver::propagate()
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

    // Calculamos o tamanho da area interna (sem bordas)
    int range_x = sx - 2 * bord;
    int range_y = sy - 2 * bord;

    dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
    // Usamos arredondamento para cima (ceil) para cobrir todo o range_x/y
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

    cudaDeviceSynchronize();

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

    cudaDeviceSynchronize();

    std::swap(device_data.pp, device_data.pc);
    std::swap(device_data.qp, device_data.qc);
}

void CudaDriver::updateHost(){
    size_t total_elements = (size_t)sx * sy * sz;
    size_t msize_vol_bytes = total_elements * sizeof(float);
    cudaMemcpy(host_data.pc, device_data.pc, msize_vol_bytes, cudaMemcpyDeviceToHost);
}

float* CudaDriver::getData(){
    return host_data.pc;
}

void CudaDriver::finalize(){
    cudaFree(device_data.ch1dxx);
    cudaFree(device_data.ch1dyy);
    cudaFree(device_data.ch1dzz);
    cudaFree(device_data.ch1dxy);
    cudaFree(device_data.ch1dyz);
    cudaFree(device_data.ch1dxz);
    cudaFree(device_data.v2px);
    cudaFree(device_data.v2pz);
    cudaFree(device_data.v2sz);
    cudaFree(device_data.v2pn);

    cudaFree(device_data.pDx);
    cudaFree(device_data.pDy);
    cudaFree(device_data.qDx);
    cudaFree(device_data.qDy);
    cudaFree(device_data.pp);
    cudaFree(device_data.pc);
    cudaFree(device_data.qp);
    cudaFree(device_data.qc);

    cudaFreeHost(host_data.vpz);
    cudaFreeHost(host_data.vsv);
    cudaFreeHost(host_data.epsilon);
    cudaFreeHost(host_data.delta);
    cudaFreeHost(host_data.phi);
    cudaFreeHost(host_data.theta);
    cudaFreeHost(host_data.pp);
    cudaFreeHost(host_data.pc);
    cudaFreeHost(host_data.qp);
    cudaFreeHost(host_data.qc);
    cudaFreeHost(host_data.ch1dxx);
    cudaFreeHost(host_data.ch1dyy);
    cudaFreeHost(host_data.ch1dzz);
    cudaFreeHost(host_data.ch1dxy);
    cudaFreeHost(host_data.ch1dyz);
    cudaFreeHost(host_data.ch1dxz);
    cudaFreeHost(host_data.v2px);
    cudaFreeHost(host_data.v2pz);
    cudaFreeHost(host_data.v2sz);
    cudaFreeHost(host_data.v2pn);
}

std::unique_ptr<Driver> createDriver(int argc, char** argv) {
    return std::make_unique<CudaDriver>(argc, argv);
}
