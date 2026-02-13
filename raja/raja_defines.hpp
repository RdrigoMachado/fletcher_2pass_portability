#pragma once

#include <RAJA/RAJA.hpp>
#include "../driver.hpp"

#if defined(RAJA_ENABLE_CUDA)

    #include <cuda_runtime.h>
    using policy = RAJA::cuda_exec<256>;

    using cuda_thread_x = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;
    using cuda_thread_y = RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>;

    using POLITICA_XY = RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
            RAJA::statement::Tile<1, RAJA::tile_fixed<BSIZE_Y>, RAJA::cuda_block_y_loop,
                RAJA::statement::Tile<0, RAJA::tile_fixed<BSIZE_X>, RAJA::cuda_block_x_loop,
                    RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                            RAJA::statement::Lambda<0>
                        >
                    >
                >
            >
        >
    >;

  #define GPU_SYNC() cudaDeviceSynchronize()
  static constexpr const char* policy_name = "CUDA";

#elif defined(RAJA_ENABLE_HIP)

    #include <hip/hip_runtime.h>
    using policy = RAJA::hip_exec<256>;

    using POLITICA_XY = RAJA::KernelPolicy<
        RAJA::statement::Tile<1, RAJA::tile_fixed<BSIZE_Y>, RAJA::hip_thread_y_loop,
            RAJA::statement::Tile<0, RAJA::tile_fixed<BSIZE_X>, RAJA::hip_thread_x_loop,
                RAJA::statement::HipKernel<
                    RAJA::statement::For<1, RAJA::hip_thread_y_loop,
                        RAJA::statement::For<0, RAJA::hip_thread_x_loop,
                            RAJA::statement::Lambda<0>
                        >
                    >
                >
            >
        >
    >;

  #define GPU_SYNC() hipDeviceSynchronize()
  static constexpr const char* policy_name = "HIP";

#else

  using policy = RAJA::seq_exec;

  using POLITICA_XY = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  #define GPU_SYNC() ((void)0)
  static constexpr const char* policy_name = "Sequential";

#endif
