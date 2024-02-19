# please follow this if you work on Clion https://www.jetbrains.com/help/clion/cuda-projects.html#set-nvcc

cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_ARCHITECTURES "native")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75)
project(cuda LANGUAGES CUDA CXX)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Detect the operating system
if(WIN32)
    set(SFML_DIR "${CMAKE_SOURCE_DIR}/externals/SFML-2.6.0_win")
    set(CUDA_DIR "${CMAKE_SOURCE_DIR}/externals/cuda-12.2_win")
elseif(UNIX)
    # you need sfml installed using 'sudo apt-get install libsfml-dev' to 
    # use find_package
    find_package(SFML COMPONENTS graphics window system REQUIRED)

    # for cuda 11:
    set(CUDA_DIR "/usr/local/cuda-11.4")
    find_package(CUDA 11.0 REQUIRED)
    # for cuda 12:
    # set(CUDA_DIR "/usr/local/cuda-12.3")
    # find_package(CUDA 12.0 REQUIRED)

    find_package(Catch2 3 REQUIRED)
    #install with
    # cd ~
    # $ git clone https://github.com/catchorg/Catch2.git
# $ cd Catch2
# $ cmake -Bbuild -H. -DBUILD_TESTING=OFF
# $ sudo cmake --build build/ --target install
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()

include_directories("${CUDA_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/externals/tinycolormap-master/include")
# include_directories("externals/tinycolormap-master/include")
include_directories("${CMAKE_SOURCE_DIR}/externals/eigen-3.4.0")
# include_directories("${CMAKE_SOURCE_DIR}/externals/sciplot")
# include_directories("externals/eigen-3.4.0")
if(WIN32)
    include_directories("${SFML_DIR}/include")
    link_directories("${SFML_DIR}/lib")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
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
    HeaderCPP.h
    HeaderCUDA.h
    math_util.cpp
    math_util.h
    static_variables.cpp
    static_variables.h
    one_leg.cu
    one_leg.cu.h
    one_leg.cpp
    cuda_util.cu
    cuda_util.h
    several_leg.cu
    several_leg.cu.h
    several_leg.cpp
    )

# target_link_libraries(cuda sfml-graphics sfml-window sfml-system)
# target_link_libraries(cuda PRIVATE Catch2::Catch2)
target_link_libraries(cuda PRIVATE Catch2::Catch2WithMain)
target_include_directories(cuda PRIVATE "${CUDA_DIR}/include")

# Copy font file to the build directory
configure_file(${CMAKE_SOURCE_DIR}/arial.ttf ${CMAKE_BINARY_DIR}/arial.ttf COPYONLY)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cuda PRIVATE -O3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(WIN32) # Define the source directory of SFML DLLs
    set(SFML_DLL_DIR "${SFML_DIR}/bin")

    # Define a list of SFML DLLs you want to copy
    set(SFML_DLLS
            "sfml-graphics-2.dll"
            "sfml-window-2.dll"
            "sfml-system-2.dll"
            # Add more DLLs if needed
    )

    # Copy the SFML DLLs to the binary directory when building
    foreach(DLL ${SFML_DLLS})
        add_custom_command(TARGET cuda POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SFML_DLL_DIR}/${DLL}"
                $<TARGET_FILE_DIR:cuda>)
    endforeach()
endif ()

include(CTest)
include(Catch)
catch_discover_tests(cuda)
