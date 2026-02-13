# Cria a biblioteca estática para o backend SYCL
add_library(sycl_backend STATIC sycl_driver.cpp)

# Adiciona o diretório atual aos includes da biblioteca
target_include_directories(sycl_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
