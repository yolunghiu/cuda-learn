cmake_minimum_required(VERSION 3.16)
project(cuda-learn CUDA CXX C)


add_executable(test test.cpp)

add_executable(01 01_hello_world.cu)
add_executable(02 02_check_dimension.cu)
add_executable(03 03_grid_block.cu)
add_executable(04 04_sum_arrays.cu)
