cmake_minimum_required(VERSION 3.18)
project(fletcher_wave VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Escolha do backend
set(BACKEND "KOKKOS" CACHE STRING "Backend to use: KOKKOS, CUDA, SYCL, RAJA, HIP")

# Cria o executável primeiro
add_executable(fletcher_${BACKEND})

# Fontes comuns (Independentes do backend)
target_sources(fletcher_${BACKEND}
    PRIVATE
        main.cpp
        model.cpp
        walltime.cpp
        source.cpp
        utils.cpp
)

# Includes comuns
target_include_directories(fletcher_${BACKEND}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Configuração Específica dos Backends ---

if(BACKEND STREQUAL "KOKKOS")
    find_package(Kokkos REQUIRED)

    # Adiciona fontes do Kokkos
    target_sources(fletcher_${BACKEND} PRIVATE kokkos/kokkos_driver.cpp)
    target_include_directories(fletcher_${BACKEND} PRIVATE kokkos)

    # Definições e Linkagem
    target_compile_definitions(fletcher_${BACKEND} PRIVATE USE_KOKKOS)
    target_link_libraries(fletcher_${BACKEND} PRIVATE Kokkos::kokkos)

elseif(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)

    # Define arquitetura padrão se não for passada via linha de comando
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89) # Default para Ada Lovelace
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(cuda)
    target_compile_definitions(fletcher_${BACKEND} PRIVATE USE_CUDA)
    target_link_libraries(fletcher_${BACKEND}
        PRIVATE
            cuda_backend
            CUDA::cudart
    )

    target_include_directories(fletcher_${BACKEND} PRIVATE cuda)

elseif(BACKEND STREQUAL "SYCL")
    # Usa a configuração do subdiretório 'sycl'
    add_subdirectory(sycl)

    # Apenas linka a biblioteca criada no subdiretório
    # NÃO adicione sycl_driver.cpp aqui novamente!
    target_link_libraries(fletcher_${BACKEND} PRIVATE sycl_backend stdc++ m)

    # Garante que o include dir do SYCL esteja visível
    target_include_directories(fletcher_${BACKEND} PRIVATE sycl)
    target_compile_definitions(fletcher_${BACKEND} PRIVATE USE_SYCL)

elseif(BACKEND STREQUAL "RAJA")
    # 1. Habilita CUDA (necessário para RAJA na GPU)
    enable_language(CUDA)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()

    # 2. Definições de caminhos (Se não forem passados via linha de comando)
    # Tenta usar variáveis de ambiente se as do CMake estiverem vazias
    if(NOT DEFINED RAJA_DIR)
        set(RAJA_DIR $ENV{RAJA_ROOT}/lib/cmake/raja)
    endif()
    if(NOT DEFINED camp_DIR)
        set(camp_DIR $ENV{CAMP_ROOT}/lib/cmake/camp)
    endif()
    if(NOT DEFINED umpire_DIR)
        set(umpire_DIR $ENV{UMPIRE_ROOT}/lib/cmake/umpire)
    endif()

    # 3. Encontra dependências na ordem correta
    find_package(camp REQUIRED)   # RAJA depende do camp
    find_package(umpire REQUIRED) # RAJA depende do umpire (conforme seu singularity)
    find_package(RAJA REQUIRED)

    # 4. Adiciona subdiretório e definições
    add_subdirectory(raja)
    target_compile_definitions(fletcher_${BACKEND} PRIVATE USE_RAJA)

    # 5. Linkagem (Agora segura)
    # Verifica se o target RAJA::RAJA existe. Se não, tenta apenas RAJA.
    if(TARGET RAJA::RAJA)
        target_link_libraries(fletcher_${BACKEND} PRIVATE raja_backend RAJA::RAJA)
    elseif(TARGET RAJA)
        target_link_libraries(fletcher_${BACKEND} PRIVATE raja_backend RAJA)
    else()
        message(FATAL_ERROR "RAJA found but target RAJA::RAJA or RAJA not found!")
    endif()

    # Linka Umpire e Camp explicitamente se necessário (segurança)
    target_link_libraries(fletcher_${BACKEND} PRIVATE umpire camp)

    target_include_directories(fletcher_${BACKEND} PRIVATE raja)

elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)

    # Defina a arquitetura HIP se necessario (ex: gfx90a para MI250X)
    # Se nao definido, o compilador tenta detectar ou compila genericamente
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()

    add_subdirectory(hip)

    target_compile_definitions(fletcher_${BACKEND} PRIVATE USE_HIP)

    target_link_libraries(fletcher_${BACKEND}
        PRIVATE
            hip_backend
            hip::host
    )

    target_include_directories(fletcher_${BACKEND} PRIVATE hip)

else()
    message(FATAL_ERROR "Backend desconhecido: ${BACKEND}")
endif()

message(STATUS "Configurando para backend: ${BACKEND}")
