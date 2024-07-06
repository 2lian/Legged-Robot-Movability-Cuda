#!/bin/bash
set -e -o pipefail
# export CXXFLAGS="-isystem /externals/eigen-3.4.0"
# cmake -DCMAKE_CXX_FLAGS=-I/externals/eigen-3.4.0 .
echo -e "\e[1;31mBUILD\e[0m"
# LDFLAGS="-Wl,--no-as-needed"
# cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
# cmake --build . -- -j8
cmake . 
# LDFLAGS="-Wl,--no-as-needed"
make -j8 
echo -e "\e[1;31mEXECUTING\e[0m"
./bench 
echo -e "\e[1;31mAFTER\e[0m"
python3 ./benchIllu.py
echo -e "\e[1;31mFINISHED -------------------------------------\e[0m"
# feh -R 1 reachability_result.png
