#include "HeaderCPP.h"
#import "HeaderCUDA.h"
#import "one_leg.cu.h"

__device__ void rotateInPlace(float3& point, float z_rot, float& cos_memory,
                              float& sin_memory) {
    sincosf(z_rot, &sin_memory, &cos_memory);
    float buffer = point.x * sin_memory;
    point.x = point.x * cos_memory - point.y * sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ void unrotateInPlace(float3& point, float z_rot, float& cos_memory,
                                float& sin_memory) {
    float buffer = point.x * -sin_memory;
    point.x = point.x * cos_memory - point.y * -sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ bool reachable_leg(float3 target, float3 body_pos,
                              LegDimensions dim) {
    float cos_memory;
    float sin_memory;
    target.x -= body_pos.x;
    target.y -= body_pos.y;
    target.z -= body_pos.z;
    rotateInPlace(target, -dim.body_angle, cos_memory, sin_memory);
    return reachability_vect(target, dim);
};

__global__ void valid_positions(Array<float3> body_map, Array<float3> target_map, LegDimensions dim,
                                Array<bool> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < body_map.length; i += stride) {
        output.elements[i] = reachable_leg(body_map.elements[i], dim);
    }
};
