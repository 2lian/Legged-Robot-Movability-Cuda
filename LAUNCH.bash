# export CXXFLAGS="-isystem /externals/eigen-3.4.0"
# cmake -DCMAKE_CXX_FLAGS=-I/externals/eigen-3.4.0 .
cmake .
make
./cuda
python3 array_vizu_1D.py
imgcat graph.png
