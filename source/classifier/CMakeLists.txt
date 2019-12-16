cmake_minimum_required(VERSION 3.15)
project(spp)


set(OPENMP3_ROOT "libraries/openmp3")

set(CMAKE_CXX_STANDARD 14)

# openmp3
file(GLOB OPENMP3_SRC
        "${OPENMP3_ROOT}/src/*.h"
        "${OPENMP3_ROOT}/src/*.cpp"
        )
link_directories(libraries/openmp3/ libraries/openmp3/src/)

# Torch
set(CMAKE_PREFIX_PATH /lib/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
        spp main.cpp
        ${OPENMP3_ROOT}/include/openmp3.h
        ${OPENMP3_ROOT}/openmp3.cpp
        ${OPENMP3_ROOT}/openmp3.h
        RCNN.cpp RCNN.hpp train.hpp train.cpp data.hpp data.cpp test.cpp test.hpp input_parser.cpp input_parser.hpp)

target_link_libraries(spp "${TORCH_LIBRARIES}")
set_property(TARGET spp PROPERTY CXX_STANDARD 14)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET spp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:spp>)
endif (MSVC)