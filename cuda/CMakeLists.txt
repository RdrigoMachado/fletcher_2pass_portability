# cuda/CMakeLists.txt

# Encontra o Toolkit do CUDA (para garantir que temos as libs)
find_package(CUDAToolkit REQUIRED)

# Cria a biblioteca
add_library(cuda_backend STATIC
    cuda_driver.cpp
)

# --- CORREÇÃO CRÍTICA ---
# Força o CMake a usar o compilador NVCC para este arquivo .cpp
# Sem isso, ele usa g++ e não acha o cuda_runtime.h nem entende <<< >>>
set_source_files_properties(cuda_driver.cpp PROPERTIES LANGUAGE CUDA)

# Linka com a biblioteca de Runtime do CUDA (cudart)
# Isso adiciona automaticamente os caminhos de include necessários
target_link_libraries(cuda_backend
    PUBLIC
    CUDA::cudart
)

# Garante que o diretório atual esteja no include path
target_include_directories(cuda_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# Otimizações (Opcional, mas recomendado)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(cuda_backend PRIVATE -O3)
endif()
