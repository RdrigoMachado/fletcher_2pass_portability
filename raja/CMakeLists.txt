add_library(raja_backend STATIC
    raja_driver.cpp
)

# Decide qual compilador usar para o arquivo .cpp do RAJA
if(RAJA_PLATFORM STREQUAL "CUDA")
    set_source_files_properties(raja_driver.cpp PROPERTIES LANGUAGE CUDA)
    # Habilita lambdas estendidos para NVIDIA
    target_compile_options(raja_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

elseif(RAJA_PLATFORM STREQUAL "HIP")
    set_source_files_properties(raja_driver.cpp PROPERTIES LANGUAGE HIP)
    # Flags HIP se necessário
    target_compile_options(raja_backend PRIVATE -O3)
endif()

# Linkagem
if(TARGET RAJA::RAJA)
    target_link_libraries(raja_backend PUBLIC RAJA::RAJA)
elseif(TARGET RAJA)
    target_link_libraries(raja_backend PUBLIC RAJA)
endif()

target_link_libraries(raja_backend PUBLIC umpire camp)

target_include_directories(raja_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
