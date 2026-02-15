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
    # 1. Define padrão se não for passado
    if(NOT DEFINED RAJA_PLATFORM)
        set(RAJA_PLATFORM "CUDA")
    endif()

    # 2. Ativação Condicional de Linguagem
    if(RAJA_PLATFORM STREQUAL "CUDA")
        # --- Configuração NVIDIA ---

        # Correção BLT (Host Compiler) para NVIDIA
        if(NOT CMAKE_CUDA_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        endif()

        enable_language(CUDA)

        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 89)
        endif()

        add_compile_definitions(RAJA_ENABLE_CUDA)

    elseif(RAJA_PLATFORM STREQUAL "HIP")
        # --- Configuração AMD ---
        enable_language(HIP)

        # Define arquitetura se não passada (native = detecta a placa atual)
        if(NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES "native")
        endif()

        add_compile_definitions(RAJA_ENABLE_HIP)
    else()
        message(FATAL_ERROR "RAJA_PLATFORM must be CUDA or HIP")
    endif()

    # 3. Encontrar Dependências (Camp, Umpire, RAJA)
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

    find_package(camp REQUIRED)
    find_package(umpire REQUIRED)
    find_package(RAJA REQUIRED)

    # 4. Adiciona Subdiretório
    add_subdirectory(raja)
    target_compile_definitions(fletcher_${BACKEND} PRIVATE USE_RAJA)

    # 5. Linkagem Segura
    if(TARGET RAJA::RAJA)
        target_link_libraries(fletcher_${BACKEND} PRIVATE raja_backend RAJA::RAJA)
    elseif(TARGET RAJA)
        target_link_libraries(fletcher_${BACKEND} PRIVATE raja_backend RAJA)
    endif()

    # Linka dependências explicitas e runtime do HIP se necessário
    target_link_libraries(fletcher_${BACKEND} PRIVATE umpire camp)

    if(RAJA_PLATFORM STREQUAL "HIP")
        target_link_libraries(fletcher_${BACKEND} PRIVATE hip::host)
    endif()

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
    )

    target_include_directories(fletcher_${BACKEND} PRIVATE hip)

else()
    message(FATAL_ERROR "Backend desconhecido: ${BACKEND}")
endif()

message(STATUS "Configurando para backend: ${BACKEND}")
