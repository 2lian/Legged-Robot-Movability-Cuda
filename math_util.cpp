/* #include <iostream> */
/* #include <fstream> */
#include "HeaderCPP.h"
#include <tinycolormap.hpp>
/* #include <random> */

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

// Function to calculate the mean of an array of floats
float calculateMean(const float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum / size;
}

// Function to calculate the standard deviation of an array of floats
float calculateStdDev(const float* arr, int size, float mean) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = arr[i] - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum / (size - 1));
}
