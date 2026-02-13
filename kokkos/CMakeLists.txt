add_library(kokkos_backend
    kokkos_driver.cpp
)

target_include_directories(kokkos_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kokkos_backend
    PUBLIC Kokkos::kokkos
)
