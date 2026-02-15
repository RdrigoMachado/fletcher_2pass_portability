#!/bin/bash
set -e # Para o script se houver erro

# --- Configurações ARM ---
ARCH="89" # L40S (Ada Lovelace)
BIN_DIR="../binaries/nvidia_arm"
BUILD_DIR="build_arm"

# Ajuste estes caminhos conforme a instalação na imagem ARM
ACPP_BIN="/opt/adaptivecpp/bin/acpp"
CAMP_PATH="/opt/camp/lib/cmake/camp"
UMPIRE_PATH="/opt/umpire/lib/cmake/umpire"
RAJA_PATH="/opt/raja/lib/cmake/raja"

mkdir -p $BIN_DIR
mkdir -p $BUILD_DIR

echo "=========================================================="
echo "   Iniciando Compilação Total - NVIDIA ARM (L40S Grace)"
echo "=========================================================="

# 1. CUDA Nativo
echo -e "\n>> [1/5] Compilando CUDA Nativo..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
cmake ../.. \
    -D BACKEND=CUDA \
    -D CMAKE_CUDA_ARCHITECTURES=$ARCH \
    -D CMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp fletcher_CUDA $BIN_DIR/
cd ..

# 2. Kokkos (Backend CUDA)
echo -e "\n>> [2/5] Compilando Kokkos (CUDA)..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
cmake ../.. \
    -D BACKEND=KOKKOS \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_ADA89=ON \
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
    -D CMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp fletcher_KOKKOS $BIN_DIR/
cd ..

# 3. SYCL (AdaptiveCpp -> CUDA)
echo -e "\n>> [3/5] Compilando SYCL (AdaptiveCpp)..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
export ACPP_TARGETS="cuda:sm_$ARCH"
cmake ../.. \
    -D BACKEND=SYCL \
    -D CMAKE_CXX_COMPILER=$ACPP_BIN \
    -D CMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp fletcher_SYCL $BIN_DIR/
cd ..

# 4. RAJA (Backend CUDA)
echo -e "\n>> [4/5] Compilando RAJA (CUDA)..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
cmake ../.. \
    -D BACKEND=RAJA \
    -D RAJA_PLATFORM=CUDA \
    -D CMAKE_CUDA_ARCHITECTURES=$ARCH \
    -D CMAKE_BUILD_TYPE=Release \
    -D camp_DIR=$CAMP_PATH \
    -D umpire_DIR=$UMPIRE_PATH \
    -D RAJA_DIR=$RAJA_PATH \
    -D CMAKE_CUDA_HOST_COMPILER=$(which g++)
make -j$(nproc)
cp fletcher_RAJA $BIN_DIR/
cd ..

# 5. HIP (Rodando sobre NVIDIA ARM)
# AVISO: Isso pode falhar se não houver pacote hip-runtime-nvidia para aarch64
echo -e "\n>> [5/5] TENTANDO compilar HIP (sobre NVIDIA ARM)..."
# Usamos um subshell (...) ou set +e para não matar o script se este passo falhar
(
    set -e
    rm -rf $BUILD_DIR/* && cd $BUILD_DIR
    export HIP_PLATFORM=nvidia
    cmake ../.. \
        -D BACKEND=HIP \
        -D CMAKE_CXX_COMPILER=hipcc \
        -D CMAKE_HIP_ARCHITECTURES=$ARCH \
        -D CMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cp fletcher_HIP $BIN_DIR/fletcher_HIP_NVIDIA
) || echo "⚠️ AVISO: Falha ao compilar HIP em ARM. Verifique se o hip-runtime-nvidia está instalado."

cd ..

echo "----------------------------------------------------------"
echo "✅ Binários ARM gerados em: $BIN_DIR"
ls -lh $BIN_DIR
