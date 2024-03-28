#pragma once
#include "HeaderCUDA.h"
#include "cuda_util.cuh"

__device__ inline bool in_sphere(const float& radius,
                                 const float3& sphere_center,
                                 const float3& target) {
    return norm3df(sphere_center.x - target.x, sphere_center.y - target.y,
                   sphere_center.z - target.z) < radius;
}

__device__ inline bool in_cylinder(const float& radius, const float& plus_z,
                                   const float& minus_z,
                                   const float3& cyl_center,
                                   const float3& target) {
    float distz = target.z - cyl_center.z;

    bool plus_condition = distz < plus_z;
    bool minus_condition = distz > minus_z;
    bool radial_condition =
        norm3df(target.x - cyl_center.x, target.y - cyl_center.y, 0) < radius;
    return radial_condition and plus_condition and minus_condition;
}

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
void launch_optimized_mem_in_cylinder(float3* centers, const size_t Nc,
                                      float3* targets, const size_t Nt,
                                      unsigned char* output, const float radius,
                                      const float plus_z, const float minus_z);
__global__ void in_sphere_cccl_kernel(float3* centers, const size_t Nc,
                                      float3* targets, const size_t Nt,
                                      int* output, const float radius);
__global__ void in_cylinder_rec(float3* centers, const size_t Nc,
                                float3* targets, const size_t Nt, int* output,
                                const float radius, const float plus_z,
                                const float minus_z);
void launch_optimized_mem_in_sphere(float3* centers, const size_t Nc,
                                    float3* targets, const size_t Nt,
                                    unsigned char* output, const float radius);

struct CylinderFunctor {
    const float radius;
    const float plus_z;
    const float minus_z;

    __host__ __device__ CylinderFunctor(float _radius, float _plus_z,
                                        float _minus_z);

    __device__ int operator()(float3 body, float3 target) const;
    __device__ bool apply(float3 body, float3 target) const;
};
