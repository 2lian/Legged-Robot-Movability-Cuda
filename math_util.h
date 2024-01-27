#pragma once
#include <Eigen/Dense>
#include <vector>

int calculateMedian(const Eigen::VectorXi& data);
float calculateMean(const float* arr, int size);
float calculateStdDev(const float* arr, int size, float mean);
