//
// Created by 2lian on 2023-09-16.
//
#pragma once
#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H
#include <driver_types.h>

int thisiscuda();

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

struct RobotDimensions {
public:
    float pI;
    float body;
    float coxa_angle_deg;
    float coxa_length;
    float tibia_angle_deg;
    float tibia_length;
    float tibia_length_squared;
    float femur_angle_deg;
    float femur_length;
    float max_angle_coxa;
    float min_angle_coxa;
    float max_angle_coxa_w_margin;
    float min_angle_coxa_w_margin;
    float max_angle_tibia;
    float min_angle_tibia;
    float max_angle_femur;
    float min_angle_femur;
    float max_angle_femur_w_margin;
    float min_angle_femur_w_margin;
    float max_tibia_to_gripper_dist;
    float positiv_saturated_femur[2];
    float negativ_saturated_femur[2];
    float fem_tib_min_host[2];
    float min_tibia_to_gripper_dist;
    float middle_TG;
};

class AutoEstimator {
private:
    cudaError_t cudaStatus;
public:
    int screenWidth;
    int screenHeight;
    int rows;
    RobotDimensions dimensions;
    Matrix table_input;
    Matrix table_input_gpu;
    Matrix result_gpu;
    Matrix result;
    Matrix result_norm;
    Matrix result_norm_gpu;
    int blockSize;
    int numBlocks;
    bool verbose;
    unsigned char* virdisTexture_gpu;
    unsigned char* virdisTexture;

    AutoEstimator(int pxWidth, int pxHeight);
    // Declare other member functions here.
    void input_as_grid();
    void change_zvalue(float zvalue);
    void alocate_gpu_mem();
    void copy_input_cpu2gpu();
    void setup_kernel();
    void compute_dist();
    void compute_result_norm();
    void AutoEstimator::convert_to_virdis();
    void copy_output_gpu2cpu();
    void delete_all();
    void error_check();
};

#endif //CUDA_HEADER_H
