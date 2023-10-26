#include <iostream>
#include "Header.h"
#include <SFML/Graphics.hpp>
#include <tinycolormap.hpp>
#include <random>
#include <Eigen/Dense>

int CalculateMedian(const Eigen::VectorXi& data) {
    // Convert the Eigen vector to a std::vector for sorting
    std::vector<int> stdVector(data.data(), data.data() + data.size());

    // Sort the std::vector
    std::sort(stdVector.begin(), stdVector.end());

    // Calculate the median
    int median;
    size_t size = stdVector.size();
    median = stdVector[size / 2]; // not very true but I don't care

    return median;
}

int main(){
    float pixel_density = 10; // sample/pixel
    float pix_size = 1; // mm between each screen pixel
    int windowWidth   = 1920;
    int windowHeight  = 1080;
    bool runOnce = 0;
    std::cout << "let's go!";
    AutoEstimator autoe{(int)(pixel_density*windowWidth),
                        (int)(pixel_density*windowHeight),
                        1 / pix_size / pixel_density};

    autoe.verbose = false;
    sf::Time deltaTime; sf::Clock clock;
    sf::Time value_change;
    sf::Time compute;
    sf::Time copying;

    autoe.reset_image();
    for(int i=0; i<10; i++) {
        clock.restart();
        autoe.dist_to_virdis_pipelinef3();
    }
//    for(int i=0; i<10; i++) {
//        clock.restart();
//        autoe.dist_to_virdis_pipelinef3();
//        std::cout << clock.restart().asMicroseconds() << " us" << std::endl;
//    }
    for(int i=0; i<10; i++) {
        clock.restart();
        autoe.reachability_to_img_pipelinef3();
        std::cout << clock.restart().asMicroseconds() << " us" << std::endl;
    }
    std::cout << " " << std::endl;
    for(int i=0; i<10; i++) {
        clock.restart();
        autoe.reachability_to_img_pipeline_tex();
        std::cout << clock.restart().asMicroseconds() << " us" << std::endl;
    }

    autoe.delete_all();
    return 0;
}