#pragma once

#include <memory>

#if defined(USE_KOKKOS)
    #define BACKEND "Kokkos"
#elif defined(USE_RAJA)
    #define BACKEND "RAJA"
#elif defined(USE_CUDA)
    #define BACKEND "CUDA"
#elif defined(USE_HIP)
    #define BACKEND "HIP"
#elif defined(USE_SYCL)
    #define BACKEND "SYCL"
#else
    #define BACKEND "CPU"
#endif

#define MI 0.2           // stability factor to compute dt
#define SIGMA  0.75      // value of sigma on formula 7 of Fletcher's paper
#define MAX_SIGMA 10.0   // above this value, SIGMA is considered infinite; as so, vsz=0
#define BSIZE_X 32
#define BSIZE_Y 16
#define NPOP 4
#define TOTAL_X (BSIZE_X + 2*NPOP)
#define TOTAL_Y (BSIZE_Y + 2*NPOP)

class Driver {
public:
    virtual void initialize(
        const int sx_, const int sy_, const int sz_,
        const float dx_, const float dy_, const float dz_,
        const float dt_,
        const int nx_, const int ny_, const int nz_,
        const int bord_, const int absorb_) = 0;
    virtual void insertSource(int index, float value) = 0;
    virtual void propagate() = 0;
    virtual void updateHost() = 0;
    virtual void finalize() = 0;
    virtual float* getData() = 0; // Adicionei isso pois estava faltando na classe base
    virtual ~Driver() = default;  // Importante para evitar vazamento de memória
};

// A "Factory Function". Cada backend implementará sua versão desta função.
std::unique_ptr<Driver> createDriver(int argc, char** argv);
