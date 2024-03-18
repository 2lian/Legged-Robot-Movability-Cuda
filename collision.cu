#include "collision.cu.h"

__device__ inline bool in_sphere(const float radius, const float3 sphere_center,
                                 const float3 target) {
    float3 dist;
    dist.x = sphere_center.x - target.x;
    dist.y = sphere_center.y - target.y;
    dist.z = sphere_center.z - target.z;
    return radius > norm3df(dist.x, dist.y, dist.z);
}

__device__ inline bool in_cylinder(const float radius, const float plus_z,
                                   const float minus_z, const float3 cyl_center,
                                   const float3 target) {
    float3 dist;
    dist.x = cyl_center.x - target.x;
    dist.y = cyl_center.y - target.y;
    bool radial_condition = norm3df(dist.x, dist.y, 0) < radius;
    dist.z = cyl_center.z - target.z;
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

__global__ void in_sphere_cccl_kernel(float3* centers, const size_t Nc,
                                      float3* targets, const size_t Nt,
                                      int* output, const float radius) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    long maxid = Nt * Nc;
    for (long i = index; i < maxid; i += stride) {
        auto center_index = i / Nt;
        auto target_index = i % Nt;
        const float3 target = targets[target_index];
        const float3 body_pos = centers[center_index];
        if (in_sphere(radius, target, body_pos)) {
            output[center_index] = -1;
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
    long maxid = Nt * Nc;
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

__global__ void in_cylinder_rec2(const float3 center, float3* targets,
                                 const size_t Nt, int* result,
                                 const float radius, const float plus_z,
                                 const float minus_z) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t target_index = index; target_index < Nt;
         target_index += stride) {
        const float3 target = targets[target_index];
        if (in_cylinder(radius, plus_z, minus_z, center, target)) {
            // atomicMin(result, (int)(-1));
            result[0] = -1;
        }
    }
}

__global__ void in_cylinder_rec(float3* centers, const size_t Nc,
                                float3* targets, const size_t Nt, int* output,
                                const float radius, const float plus_z,
                                const float minus_z) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // int blockSize = min(256, (int)Nt);
    size_t blockSize = 256;
    size_t numBlock = (Nt + blockSize - 1) / blockSize;
    for (size_t center_index = index; center_index < Nc;
         center_index += stride) {
        const float3 body_pos = centers[center_index];
        int* result_pointer = &output[center_index];
        in_cylinder_rec2<<<numBlock, blockSize>>>(
            body_pos, targets, Nt, result_pointer, radius, plus_z, minus_z);
    }
};

__host__ __device__ CylinderFunctor::CylinderFunctor(float _radius,
                                                     float _plus_z,
                                                     float _minus_z)
    : radius(_radius), plus_z(_plus_z), minus_z(_minus_z) {}

__device__ int CylinderFunctor::operator()(float3 body, float3 target) const {
    bool r = in_cylinder(radius, plus_z, minus_z, body, target);
    return r ? -1 : 0;
}
