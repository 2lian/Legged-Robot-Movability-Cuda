#pragma once
#include "HeaderCUDA.h"

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
__device__ bool reachability_circles(const float3& point, const LegDimensions& dim);
__global__ void reachability_circles_kernel(const Array<float3> input,
                                            const LegDimensions dimensions,
                                            Array<bool> const output);
__global__ void distance_circles_kernel(const Array<float3> input,
                                        const LegDimensions dim,
                                        Array<float3> const output);
/* __global__ void dist_kernel(const Arrayf3 input, LegDimensions dimensions, */
/*                             Arrayf3 const output); */

/* __global__ void reachability_kernel(const Arrayf3 input, */
/*                                     const LegDimensions dimensions, */
/*                                     Arrayb const output); */
