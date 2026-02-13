add_library(raja_backend STATIC
    raja_driver.cpp
)

set_source_files_properties(raja_driver.cpp PROPERTIES LANGUAGE CUDA)

# Linka bibliotecas
if(TARGET RAJA::RAJA)
    target_link_libraries(raja_backend PUBLIC RAJA::RAJA)
elseif(TARGET RAJA)
    target_link_libraries(raja_backend PUBLIC RAJA)
endif()

target_link_libraries(raja_backend PUBLIC umpire camp)

target_include_directories(raja_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- CORREÇÃO IMPORTANTE ---
# Habilita lambdas estendidos (Obrigatório para RAJA com CUDA)
target_compile_options(raja_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
