#include "math_util.h"
#include "HeaderCPP.h"
#include <fstream>
#include <iostream>

int calculateMedian(const Eigen::VectorXi& data) {
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

bool close(float a, float b, float interval) {
    return (a - interval < b) && (b < a + interval);
}

// Function to save array and its length to binary file
template <typename T>
void saveArrayToFile(T* array, size_t length, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(array), length * sizeof(T));
        file.close();
    } else {
        std::cout << "error savinf file";
    }
}

template void saveArrayToFile<int>(int* array, size_t length,
                                   const char* filename);
template void saveArrayToFile<float>(float* array, size_t length,
                                   const char* filename);
