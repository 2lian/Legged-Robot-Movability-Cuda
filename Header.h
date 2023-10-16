//
// Created by 2lian on 2023-09-16.
//
#pragma once
#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H
#include <driver_types.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrixf;

typedef struct {
    int width;
    int height;
    int stride;
    bool* elements;
} Matrixb;

typedef struct {
    int length;
    float3* elements;
} Arrayf3;

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
    float positiv_saturated_femur[2];
    float negativ_saturated_femur[2];
    float fem_tib_min_host[2];
    float max_tibia_to_gripper_dist;
    float min_tibia_to_gripper_dist;
    float middle_TG;
    float middle_TG_radius;
    float middle_TG_radius_w_margin;
    float femur_overmargin;
};

class AutoEstimator {
private:
    cudaError_t cudaStatus;
public:
    int screenWidth;
    int screenHeight;
    int rows;

    RobotDimensions dimensions{};
    Matrixf table_input{};
    Matrixf table_input_gpu{};
    Matrixf result_gpu{};
    Matrixf result{};
    Matrixf result_norm{};
    Matrixf result_norm_gpu{};
    Matrixf targetset{};
    Matrixf targetset_gpu{};

    Arrayf3 arr_input{};
    Arrayf3 arr_input_gpu{};

    int blockSize;
    int numBlocks;
    bool verbose;
    unsigned char* virdisTexture_gpu{};
    unsigned char* virdisTexture;
    int* gpu_accumulator{};

    AutoEstimator(int pxWidth, int pxHeight, float scale);
    // Declare other member functions here.
    void input_as_grid(const float scale_factor) const;
    void change_z_value(float zvalue);
    void allocate_gpu_mem();
    void copy_input_cpu2gpu();
    void setup_kernel();
    void compute_dist();
    void compute_result_norm();
    void convert_to_virdis();
    void copy_output_gpu2cpu();
    void delete_all();
    void error_check();
    void virdisresult_gpu2cpu();

    void dist_to_virdis_pipeline();

    void reachability_to_img_pipeline();
    void reachability_to_img_pipelinef3();

    void switch_zy();

    void change_y_value(float value);

    void all_reachable_default_to_image();

    void compute_leg0_by_accumulation();

    void dist_to_virdis_pipelinef3();

    void derivate_output();
};

#endif //CUDA_HEADER_H
