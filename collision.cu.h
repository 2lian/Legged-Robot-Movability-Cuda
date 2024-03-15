#pragma once
#include "HeaderCUDA.h"

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

struct CylinderFunctor {
    float radius;
    float plus_z;
    float minus_z;

    __host__ __device__ CylinderFunctor(float _radius, float _plus_z,
                                        float _minus_z);

    __device__ int operator()(float3 body, float3 target) const;
};
