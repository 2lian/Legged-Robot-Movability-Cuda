cmake_minimum_required(VERSION 3.26)
project(cuda LANGUAGES CUDA CXX)

# Copy font file to the build directory
configure_file(${CMAKE_SOURCE_DIR}/arial.ttf ${CMAKE_BINARY_DIR}/arial.ttf COPYONLY)

include_directories("C:/Users/2lian/source/cuda_include")
include_directories("C:/Users/2lian/source/include")

set(SFML_INCLUDE_DIR "C:/Users/2lian/source/SFML-2.6.0/include")
set(SFML_LIBRARY_DIR "C:/Users/2lian/source/SFML-2.6.0/lib")

include_directories(${SFML_INCLUDE_DIR})
link_directories(${SFML_LIBRARY_DIR})


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

add_executable(cuda
        notcuda.cpp
        Header.h
        cuda_with_header.cu)

target_link_libraries(cuda sfml-graphics sfml-window sfml-system)

target_link_libraries(cuda PRIVATE ${SFML_LIBRARIES}) #unsure if usefull
target_include_directories(cuda PRIVATE ${SFML_INCLUDE_DIRS})
target_include_directories(cuda PRIVATE "C:/Users/2lian/source/cuda_include")

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cuda PRIVATE -O3)

# Define the source directory of SFML DLLs
set(SFML_DLL_DIR "C:/Users/2lian/source/SFML-2.6.0/bin")

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