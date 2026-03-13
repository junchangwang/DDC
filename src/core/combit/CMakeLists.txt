add_library(combit
    src/types.cpp
    src/combit_encoder.cpp
    src/combit_decoder.cpp
    src/bitmap_vector.cpp
)

target_include_directories(combit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)