#include "collision.cu.h"

__device__ bool in_sphere(const float radius, const float3 sphere_center,
                          const float3 target) {
    float3 dist;
    dist.x = sphere_center.x - target.x;
    dist.y = sphere_center.y - target.y;
    dist.z = sphere_center.z - target.z;
    return radius > norm3df(dist.x, dist.y, dist.z);
}

__device__ bool in_cylinder(const float radius, const float plus_z,
                            const float minus_z, const float3 sphere_center,
                            const float3 target) {
    float3 dist;
    dist.x = sphere_center.x - target.x;
    dist.y = sphere_center.y - target.y;
    bool radial_condition = norm3df(dist.x, dist.y, 0) < radius;
    dist.z = sphere_center.z - target.z;
    bool plus_condition = dist.z < plus_z;
    bool minus_condition = dist.z > minus_z;
    return radial_condition and plus_condition and minus_condition;
}

__global__ void in_cylinder_accu_kernel(Array<float3> centers,
                                        Array<float3> targets,
                                        Array<int> output, const float radius,
                                        const float plus_z,
                                        const float minus_z) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    long maxid = (long)centers.length * (long)targets.length;
    for (long i = index; i < maxid; i += stride) {
        int center_index = i / targets.length;
        int target_index = i % targets.length;
        const float3 target = targets.elements[target_index];
        const float3 body_pos = centers.elements[center_index];
        if (in_cylinder(radius, plus_z, minus_z, target, body_pos)) {
            output.elements[center_index] = -1;
        }
    }
};

__global__ void in_cylinder_cccl_kernel(float3* centers, const size_t Nc,
                                        float3* targets, const size_t Nt,
                                        int* output, const float radius,
                                        const float plus_z,
                                        const float minus_z) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    long maxid = (long)Nt * (long)Nc;
    for (long i = index; i < maxid; i += stride) {
        auto center_index = i / Nt;
        auto target_index = i % Nt;
        const float3 target = targets[target_index];
        const float3 body_pos = centers[center_index];
        if (in_cylinder(radius, plus_z, minus_z, target, body_pos)) {
            output[center_index] = -1;
        }
    }
};
