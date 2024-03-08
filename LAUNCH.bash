# export CXXFLAGS="-isystem /externals/eigen-3.4.0"
# cmake -DCMAKE_CXX_FLAGS=-I/externals/eigen-3.4.0 .
python3 before.py || exit
cmake . || exit
make || exit
./cuda || exit
python3 after.py || exit
echo "FINISHED"
# feh -R 1 reachability_result.png
