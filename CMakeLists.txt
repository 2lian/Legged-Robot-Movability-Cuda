cmake_minimum_required(VERSION 3.26)
project(cuda LANGUAGES CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# set(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")

# Detect the operating system
# you need sfml installed using 'sudo apt-get install libsfml-dev' to
# use find_package
# find_package(SFML COMPONENTS graphics window system REQUIRED)

# for cuda 11:
# set(CUDA_DIR "/usr/local/cuda-11.4")
# find_package(CUDA 11.0 REQUIRED)
# for cuda 12:
set(CUDA_DIR "/usr/local/cuda")
# find_package(CUDA 12.0 REQUIRED)

find_package(Catch2 3 REQUIRED)
#install with
# cd ~
# $ git clone https://github.com/catchorg/Catch2.git
# $ cd Catch2
# $ cmake -Bbuild -H. -DBUILD_TESTING=OFF
# $ sudo cmake --build build/ --target install

include_directories("${CUDA_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/externals/tinycolormap-master/include")
# include_directories("externals/tinycolormap-master/include")
include_directories("${CMAKE_SOURCE_DIR}/externals/eigen-3.4.0")
# include_directories("${CMAKE_SOURCE_DIR}/externals/sciplot")
# include_directories("externals/eigen-3.4.0")

# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CXX_STANDARD 17)
# add_executable(one_leg
#     HeaderCPP.h
#     HeaderCUDA.h
#     math_util.cpp
#     math_util.h
#     static_variables.cpp
#     static_variables.h
#     one_leg.cu
#     one_leg.cu.h
#     one_leg.cpp
#     cuda_util.cu
#     cuda_util.h
#     )

add_executable(cuda
    one_leg.cu
    one_leg.cu.h
    # one_leg.cpp
    cuda_util.cu
    cuda_util.cuh
    several_leg.cu
    several_leg.cu.h
    several_leg.cpp
    collision.cu
    collision.cu.h
    unified_math_cuda.cu
    unified_math_cuda.cu.h
    HeaderCPP.h
    HeaderCUDA.h
    math_util.cpp
    math_util.h
    static_variables.cpp
    static_variables.h
    cross_compiled.cu
    cross_compiled.cuh
    )

# target_link_libraries(cuda sfml-graphics sfml-window sfml-system)
# target_link_libraries(cuda PRIVATE Catch2::Catch2)
# target_link_libraries(cuda PRIVATE Catch2::Catch2WithMain)
target_include_directories(cuda PRIVATE "${CUDA_DIR}/include")

# Copy font file to the build directory
configure_file(${CMAKE_SOURCE_DIR}/arial.ttf ${CMAKE_BINARY_DIR}/arial.ttf COPYONLY)

set_target_properties(cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

# target_compile_options(cuda PRIVATE -O3)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -prec-div=false -prec-sqrt=false -ftz=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -pg")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

# include(CTest)
# include(Catch)
# catch_discover_tests(cuda)
