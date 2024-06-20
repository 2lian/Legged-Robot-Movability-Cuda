#pragma once
#include "HeaderCUDA.h"
#include "settings.h"
// #include "unified_math_cuda.cu.h"
typedef float4 Quaternion;

// __global__ void dist_kernel(const Array<float3> input,
//                             const LegDimensions dimensions,
//                             Array<float3> const output);
//
// __global__ void reachability_kernel(const Array<float3> input,
//                                     const LegDimensions dimensions,
//                                     Array<bool> const output);
//
// __global__ void reachability_abs_tib_kernel(const Array<float3> input,
//                                             const LegDimensions dimensions,
//                                             Array<bool> const output);
//
__global__ void forward_kine_kernel(const Array<float3> input,
                                    const LegDimensions dimensions,
                                    Array<float3> const output);

// __device__ bool reachability_vect(const float3& point,
//                                   const LegDimensions& dim);
//
// __device__ bool reachability_absolute_tibia_limit(const float3& point,
//                                                   const LegDimensions& dim);
// __device__ bool reachability_circles(const float3& point, const LegDimensions& dim);
__global__ void reachability_circles_kernel(const Array<float3> input,
                                            const LegDimensions dimensions,
                                            Array<bool> const output);
__global__ void distance_circles_kernel(const Array<float3> input,
                                        const LegDimensions dim,
                                        Array<float3> const output);
__global__ void reachability_global_kernel(const Array<float3> input,
                                           const LegDimensions dim, Array<bool> output);
__global__ void distance_global_kernel(const Array<float3> input, const LegDimensions dim,
                                       Array<float3> output);
__global__ void recursive_kernel(Box box, const Array<float3> input,
                                 const LegDimensions leg, Array<float3> output,
                                 uchar depth, float radius, bool validity);
__device__ bool distance(float3& point, const LegDimensions& dim, const Quaternion quat);
