cmake_minimum_required(VERSION 3.16)
project(cuda-learn CUDA CXX C)


add_executable(test test.cpp)

add_executable(01 01_hello_world.cu)
add_executable(02 02_check_dimension.cu)
add_executable(03 03_grid_block.cu)
add_executable(04 04_sum_arrays.cu)
add_executable(05 05_vector_add.cu)
add_executable(06 06_thread_index.cu)
add_executable(07 07_sum_matrix.cu)
add_executable(08 08_device_information.cu)
add_executable(09 09_warp_divergence.cu)
add_executable(10 10_sum_matrix2D.cu)
