#pragma once
#include "HeaderCUDA.h"

__device__ bool in_sphere(const float radius, const float3 sphere_center,
                          const float3 target);

__device__ bool in_cylinder(const float radius, const float plus_z,
                            const float minus_z, const float3 sphere_center,
                            const float3 target);

__global__ void in_cylinder_accu_kernel(Array<float3> centers,
                                        Array<float3> targets,
                                        Array<int> output, const float radius,
                                        const float plus_z,
                                        const float minus_z);

__global__ void in_cylinder_cccl_kernel(float3* centers, const size_t Nc,
                                        float3* targets, const size_t Nt,
                                        int* output, const float radius,
                                        const float plus_z,
                                        const float minus_z);
__global__ void in_cylinder_rec(float3* centers, const size_t Nc,
                                float3* targets, const size_t Nt, int* output,
                                const float radius, const float plus_z,
                                const float minus_z);
