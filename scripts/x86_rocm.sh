#!/bin/bash
set -e

# Detecta arquitetura automaticamente ou força gfx1100 se preferir
HIP_ARCH="native"
BIN_DIR="../binaries/hip_amd"
BUILD_DIR="build_hip"

mkdir -p $BIN_DIR
mkdir -p $BUILD_DIR

echo "=== Iniciando Build para AMD HIP (Arch: $HIP_ARCH) ==="

# 1. HIP Nativo
echo ">> Compilando HIP Nativo..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
cmake ../.. -DBACKEND=HIP -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp fletcher_HIP $BIN_DIR/
cd ..

# 2. Kokkos (HIP)
echo ">> Compilando Kokkos (HIP)..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
# Ajuste ARCH_VEGA90A para sua placa (ex: VEGA90A=MI200, NAVI31=RX7900) ou deixe o Kokkos auto-detectar se configurado
cmake ../.. -DBACKEND=KOKKOS -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp fletcher_KOKKOS $BIN_DIR/
cd ..

# 3. SYCL (HIP Target)
echo ">> Compilando SYCL..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
# ATENCAO: Ajuste para gfx90a (MI200) ou gfx1100 (RDNA3) conforme a maquina
export ACPP_TARGETS="hip:gfx1100"
cmake ../.. -DBACKEND=SYCL -DCMAKE_CXX_COMPILER=/opt/adaptivecpp/bin/acpp -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp fletcher_SYCL $BIN_DIR/
cd ..

# 4. RAJA (HIP)
echo ">> Compilando RAJA..."
rm -rf $BUILD_DIR/* && cd $BUILD_DIR
cmake ../.. -DBACKEND=RAJA -DRAJA_PLATFORM=HIP -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" -DCMAKE_BUILD_TYPE=Release \
    -Dcamp_DIR=/opt/camp/lib/cmake/camp -Dumpire_DIR=/opt/umpire/lib/cmake/umpire -DRAJA_DIR=/opt/raja/lib/cmake/raja
make -j$(nproc)
cp fletcher_RAJA $BIN_DIR/
cd ..

echo "✅ Sucesso! Binários em $BIN_DIR"
ls -lh $BIN_DIR
