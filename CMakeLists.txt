cmake_minimum_required(VERSION 3.31)

set(PROJECT gridwait)
PROJECT(${PROJECT} CXX)

set(CMAKE_HIP_ARCHITECTURES gfx90a)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++20 -Wall -Ofast -DNDEBUG -munsafe-fp-atomics")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Ofast -march=native -I/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include")
enable_language(HIP)

find_package(MPI REQUIRED)
find_package(spdlog REQUIRED)
add_executable(${PROJECT} main.cpp)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(${PROJECT} PUBLIC spdlog::spdlog)
