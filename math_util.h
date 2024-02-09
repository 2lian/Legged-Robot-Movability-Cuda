#pragma once
#include <Eigen/Dense>
#include <vector>

int calculateMedian(const Eigen::VectorXi& data);
float calculateMean(const float* arr, int size);
float calculateStdDev(const float* arr, int size, float mean);
bool close(float a, float b,float interval);

template<typename T>
void saveArrayToFile(T* array, size_t length, const char* filename);
