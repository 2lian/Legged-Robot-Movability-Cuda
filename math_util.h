#pragma once
#include <Eigen/Dense>
#include <vector>
#include "HeaderCUDA.h"

int calculateMedian(const Eigen::VectorXi& data);
float calculateMean(const float* arr, int size);
float calculateStdDev(const float* arr, int size, float mean);
bool close(float a, float b,float interval);

template<typename T>
void saveArrayToFile(T* array, long length, const char* filename);

template <typename T>
Array<T> readArrayFromFile(const char* filename);


template <typename T>
Array<float3> threeArrays2float3Arr(Array<T> x, Array<T> y, Array<T> z);
