#include "collision.cu.h"
#include <cstddef>

__device__ inline bool in_sphere(const float radius, const float3 sphere_center,
                                 const float3 target) {
    float3 dist;
    dist.x = sphere_center.x - target.x;
    dist.y = sphere_center.y - target.y;
    dist.z = sphere_center.z - target.z;
    return norm3df(dist.x, dist.y, dist.z) < radius;
}

__device__ inline bool in_cylinder(const float radius, const float plus_z,
                                   const float minus_z, const float3 cyl_center,
                                   const float3 target) {
    float3 dist;
    dist.x = target.x - cyl_center.x;
    dist.y = target.y - cyl_center.y;
    dist.z = target.z - cyl_center.z;

    bool radial_condition = norm3df(dist.x, dist.y, 0) < radius;
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

__global__ void in_sphere_mem_kernel(float3* centers, const size_t Nc,
                                     float3* targets, const size_t Nt,
                                     unsigned char* output,
                                     const float radius) {
    __shared__ bool colliding;
    __shared__ float3 center_value;
    auto center_index = blockIdx.x;
    auto target_index = threadIdx.x;

    if (target_index == 0) {
        colliding = false;
        center_value = centers[center_index];
    }
    __syncthreads();

    if ((center_index < Nc) and (target_index < Nt)) {
        const float3 target = targets[target_index];
        if (in_sphere(radius, center_value, target)) {
            colliding = true;
        }
    }

    __syncthreads();
    if ((target_index == 0) && (colliding)) {
        output[center_index] = 1;
    }
};

void launch_optimized_mem_in_sphere(float3* centers, const size_t Nc,
                                    float3* targets, const size_t Nt,
                                    unsigned char* output, const float radius) {
    size_t processed_index = 0;
    size_t max_block_size = 1024 / 1;
    // size_t max_block_size = 1;
    size_t numBlock = Nc;
    while (processed_index < Nt) {
        size_t targets_left_to_process = Nt - processed_index;
        size_t blockSize = std::min(targets_left_to_process, max_block_size);
        float3* sub_target_ptr = targets + processed_index;
        size_t sub_target_size = blockSize;

        // std::cout << "in_sphere_mem_kernel " << numBlock << " " << blockSize
        // << std::endl;
        in_sphere_mem_kernel<<<numBlock, blockSize>>>(
            centers, Nc, sub_target_ptr, sub_target_size, output, radius);

        processed_index += blockSize;
    }
}

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
        if (in_cylinder(radius, plus_z, minus_z, body_pos, target)) {
            output[center_index] = -1;
        }
    }
};

__global__ void in_cylinder_mem_kernel(float3* centers, const size_t Nc,
                                       float3* targets, const size_t Nt,
                                       unsigned char* output,
                                       const float radius, const float plus_z,
                                       const float minus_z) {
    __shared__ bool colliding;
    __shared__ float3 body_pos;
    auto center_index = blockIdx.x;
    auto target_index = threadIdx.x;

    if (target_index == 0) {
        colliding = false;
        body_pos = centers[center_index];
    }
    __syncthreads();

    if ((center_index < Nc) and (target_index < Nt)) {
        const float3 target = targets[target_index];
        if (in_cylinder(radius, plus_z, minus_z, body_pos, target)) {
            colliding = true;
        }
    }

    __syncthreads();
    if ((target_index == 0) && (colliding)) {
        output[center_index] = 1;
    }
};

void launch_optimized_mem_in_cylinder(float3* centers, const size_t Nc,
                                      float3* targets, const size_t Nt,
                                      unsigned char* output, const float radius,
                                      const float plus_z, const float minus_z) {
    size_t processed_index = 0;
    size_t max_block_size = 1024 / 1;
    // size_t max_block_size = 1;
    size_t numBlock = Nc;
    while (processed_index < Nt) {
        size_t targets_left_to_process = Nt - processed_index;
        size_t blockSize = std::min(targets_left_to_process, max_block_size);
        float3* sub_target_ptr = targets + processed_index;
        size_t sub_target_size = blockSize;

        in_cylinder_mem_kernel<<<numBlock, blockSize>>>(
            centers, Nc, sub_target_ptr, sub_target_size, output, radius,
            plus_z, minus_z);

        processed_index += blockSize;
    }
}

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

struct InOutCylinderFunctor {
    float radius_out;
    float plus_z_out;
    float minus_z_out;
    float radius_in;
    float plus_z_in;
    float minus_z_in;

    __host__ __device__ InOutCylinderFunctor(float _radius_out,
                                             float _plus_z_out,
                                             float _minus_z_out,
                                             float _radius_in, float _plus_z_in,
                                             float _minus_z_in)
        : radius_out(_radius_out), plus_z_out(_plus_z_out),
          minus_z_out(_minus_z_out), radius_in(_radius_in),
          plus_z_in(_plus_z_in), minus_z_in(_minus_z_in) {}

    __device__ bool operator()(float3 body, float3 target) const {
        bool inside_inside_cyl =
            in_cylinder(radius_in, plus_z_in, minus_z_in, body, target);
        bool inside_outside_cyl =
            in_cylinder(radius_out, plus_z_out, minus_z_out, body, target);
        return inside_inside_cyl and (not inside_outside_cyl);
    };
};
