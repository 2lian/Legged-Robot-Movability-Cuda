#pragma once
#include "HeaderCPP.h"
#include <driver_types.h>
#include "cuda_runtime_api.h"

__device__ constexpr float pIgpu =
    3.14159265358979323846264338327950288419716939937510582097f;

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
    float* elements;
} Arrayf;

typedef struct {
    int length;
    float3* elements;
} Arrayf3;

typedef struct {
    int length;
    bool* elements;
} Arrayb;

class AutoEstimator {
  private:
    cudaError_t cudaStatus;

  public:
    int screenWidth;
    int screenHeight;
    int rows;

    LegDimensions dimensions{};
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
