# export CXXFLAGS="-isystem /externals/eigen-3.4.0"
# cmake -DCMAKE_CXX_FLAGS=-I/externals/eigen-3.4.0 .
rm "timing_results.txt"
python3 before.py || exit
# LDFLAGS="-Wl,--no-as-needed"
cmake --build . -- -j8 || exit
# cmake . || exit
# LDFLAGS="-Wl,--no-as-needed"
make -j8 || exit
./cuda || exit
python3 after.py || exit
echo "FINISHED"
# feh -R 1 reachability_result.png
