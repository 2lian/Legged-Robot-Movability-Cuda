# please follow this if you work on Clion https://www.jetbrains.com/help/clion/cuda-projects.html#set-nvcc

cmake_minimum_required(VERSION 3.26)
project(cuda LANGUAGES CUDA CXX)

# Detect the operating system
if(WIN32)
    set(SFML_DIR "${CMAKE_SOURCE_DIR}/externals/SFML-2.6.0_win")
    set(CUDA_DIR "${CMAKE_SOURCE_DIR}/externals/cuda-12.2_win")
elseif(UNIX)
    # you need sfml installed using 'sudo apt-get install libsfml-dev' to use find_package
    find_package(SFML COMPONENTS graphics window system REQUIRED)
    set(CUDA_DIR "${CMAKE_SOURCE_DIR}/externals/cuda-12.2_linux")
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()

include_directories("${CUDA_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/externals/tinycolormap-master/include")
include_directories("${CMAKE_SOURCE_DIR}/externals/eigen-3.4.0")
if(WIN32)
    include_directories("${SFML_DIR}/include")
    link_directories("${SFML_DIR}/lib")
endif()

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda
#        openGL_viz.cpp
#        Header.h
#        cuda_using_textures.cu
        create_virdis_tex.cu
        #        cuda_mess.cu
)

target_link_libraries(cuda sfml-graphics sfml-window sfml-system)
target_include_directories(cuda PRIVATE "${CUDA_DIR}/include")

# Copy font file to the build directory
configure_file(${CMAKE_SOURCE_DIR}/arial.ttf ${CMAKE_BINARY_DIR}/arial.ttf COPYONLY)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cuda PRIVATE -O3)

if(WIN32)
    # Define the source directory of SFML DLLs
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