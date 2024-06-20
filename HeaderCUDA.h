#pragma once
#include "HeaderCPP.h"
#include "cuda_runtime_api.h"
#include <cstddef>
#include <driver_types.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
// #include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <cuda_profiler_api.h>

// #include <thrust/device_vector.h>
// #include <cub/cub.cuh>
// #include <cuda/std/atomic>
#define PI 3.14159265358979323846264338327950288419716939937510582097f
typedef unsigned char uchar;
__device__ constexpr float pIgpu =
    3.14159265358979323846264338327950288419716939937510582097f;


typedef struct PlaneImage {
    float3 normal;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
} PlaneImage;

typedef struct Box {
    float3 center;
    float3 topOffset;
    // float3 bottom;
} Box;

template <typename T> struct Array {
    size_t length;
    T* elements;
};

template <> struct Array<float3> {
    size_t length;
    float3* elements;
};

template <> struct Array<float> {
    size_t length;
    float* elements;
};

template <> struct Array<bool> {
    size_t length;
    bool* elements;
};

template <> struct Array<int> {
    size_t length;
    int* elements;
};

template <> struct Array<LegDimensions> {
    size_t length;
    LegDimensions* elements;
};

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
//
// // Macro for timing measurements
// #define CUDA_TIMING_INIT()                                                    \
//     cudaEvent_t* start;                                                        \
//     cudaEvent_t* stop;
//
// #define CUDA_TIMING_START()                                                    \
//     start = new cudaEvent_t;                                                   \
//     stop = new cudaEvent_t;                                                    \
//     cudaEventCreate(start);                                                    \
//     cudaEventCreate(stop);                                                     \
//     cudaEventRecord(*start);
//
// #define CUDA_TIMING_STOP(label)                                                \
//     {                                                                          \
//         cudaEventRecord(*stop);                                                \
//         cudaEventSynchronize(*stop);                                           \
//         float milliseconds = 0;                                                \
//         cudaEventElapsedTime(&milliseconds, *start, *stop);                    \
//         cudaEventDestroy(*start);                                              \
//         cudaEventDestroy(*stop);                                               \
//         {                                                                      \
//             FILE* file = fopen("timing_results.txt", "a");                     \
//             if (file) {                                                        \
//                 fprintf(file, "[%s] Elapsed time: %.2f ms\n", label,           \
//                         milliseconds);                                         \
//                 fclose(file);                                                  \
//             }                                                                  \
//         }                                                                      \
//     }                                                                          \
//     delete start;                                                              \
//     delete stop;

// empty macro
#define CUDA_TIMING_INIT()

#define CUDA_TIMING_START()

#define CUDA_TIMING_STOP(label)
