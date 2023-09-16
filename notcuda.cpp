#include <iostream>
#include "Header.h"

int maincpp() {
    std::cout << "hey that's c++" << std::endl;
    thisiscuda();
    return 0;
}

int main(void){
    AutoEstimator autoe{6, 7};
    autoe.delete_all();
}