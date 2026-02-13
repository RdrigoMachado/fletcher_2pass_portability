# hip/CMakeLists.txt

# Habilita HIP
enable_language(HIP)

# Cria a biblioteca estatica
add_library(hip_backend STATIC
    hip_driver.cpp
)

# Inclui diretorio atual
target_include_directories(hip_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# Opcoes de compilacao HIP
# -O3: Otimizacao
# --offload-arch=native: Tenta detectar a GPU do sistema (ou defina explicitamente no CMake raiz)
target_compile_options(hip_backend
    PRIVATE
    -O3 -ffast-math
)
