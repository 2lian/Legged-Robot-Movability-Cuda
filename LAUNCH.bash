#!/bin/bash
set -e -o pipefail
# export CXXFLAGS="-isystem /externals/eigen-3.4.0"
# cmake -DCMAKE_CXX_FLAGS=-I/externals/eigen-3.4.0 .
rm "timing_results.txt" || true
echo -e "\e[1;31mBEFORE\e[0m"
python3 before.py 
echo -e "\e[1;31mBUILD\e[0m"
# LDFLAGS="-Wl,--no-as-needed"
cmake --build . -- -j8 
# cmake . 
# LDFLAGS="-Wl,--no-as-needed"
make -j8 
echo -e "\e[1;31mEXECUTING\e[0m"
./cuda 
echo -e "\e[1;31mAFTER\e[0m"
python3 after.py 
echo -e "\e[1;31mFINISHED -------------------------------------\e[0m"
# feh -R 1 reachability_result.png
