#pragma once

#include <cmath>

// ============================================================
// Backend-inline macro
// ============================================================
#if defined(USE_SYCL) || defined(SYCL_LANGUAGE_VERSION)
  #define INLINE inline
#elif defined(USE_KOKKOS)
  #include <Kokkos_Core.hpp>
  #define INLINE KOKKOS_INLINE_FUNCTION
#elif defined(USE_RAJA)
  #include <RAJA/RAJA.hpp>
  #define INLINE RAJA_HOST_DEVICE RAJA_INLINE
#elif defined(__CUDACC__) || defined(__HIPCC__) || defined(USE_CUDA) || defined(USE_HIP)
  #define INLINE __host__ __device__ inline
#else
  #define INLINE inline
#endif

// ============================================================
// 3D -> 1D index helper
// ============================================================
INLINE int ind(int ix, int iy, int iz, int sx, int sy)
{
  return (iz * sy + iy) * sx + ix;
}

// ============================================================
// Accessor abstraction
// ============================================================
template<typename T>
struct accessor;

// --------------------
// Raw pointer backend (Mutable)
// --------------------
template<>
struct accessor<float*> {
  INLINE static float& get(float* v, int i) {
    return v[i];
  }
  // Sobrecarga para permitir passar const ptr para o accessor mutavel (opcional mas util)
  INLINE static const float& get(const float* v, int i) {
    return v[i];
  }
};

// --------------------
// Raw pointer backend (Const) - CORREÇÃO CRITICA AQUI
// --------------------
// Necessário porque Der1 recebe 'const float*'
template<>
struct accessor<const float*> {
  INLINE static const float& get(const float* v, int i) {
    return v[i];
  }
};

#ifdef USE_KOKKOS
// --------------------
// Kokkos View backend
// --------------------
template<typename MemSpace>
struct accessor<Kokkos::View<float*, MemSpace>> {
  INLINE static float& get(const Kokkos::View<float*, MemSpace>& v, int i) {
    return v(i);
  }
};
#endif
