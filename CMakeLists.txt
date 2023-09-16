cmake_minimum_required(VERSION 3.26)
project(cuda LANGUAGES CUDA CXX)
#As an example, you can write project(project_name LANGUAGES CUDA CXX) to enable both CUDA and C++ as your project languages.

#To specify compiler flags for NVCC, set the CMAKE_CUDA_FLAGS variable:
#set(CMAKE_CUDA_FLAGS "-Wall")
#This way, the flags will be used globally for all targets.
#Another approach is to set the flags for specific targets with the target_compile_options command. For instance:
#
#target_compile_options(
#my_target PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#--generate-line-info>)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda cuda_main.cu
        new_cuda_source.cu
        notcuda.cpp
        Header.h)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
