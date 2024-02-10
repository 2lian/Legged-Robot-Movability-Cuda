# export CXXFLAGS="-isystem /externals/eigen-3.4.0"
# cmake -DCMAKE_CXX_FLAGS=-I/externals/eigen-3.4.0 .
cmake .
python3 before.py
make
./cuda
python3 array_vizu_1D.py
# feh -A --auto-reload=1 graph.png
