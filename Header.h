//
// Created by 2lian on 2023-09-16.
//
#pragma once
#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

int thisiscuda();

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
    int blockSize;
    int numBlocks;

    AutoEstimator(int pxWidth, int pxHeight);
    // Declare other member functions here.
    void input_as_grid();
    void change_zvalue(float zvalue);
    void alocate_gpu_mem();
    void copy_input_cpu2gpu();
    void setup_kernel();
    void compute_dist();
    void copy_output_gpu2cpu();
    void delete_all();
};

#endif //CUDA_HEADER_H
