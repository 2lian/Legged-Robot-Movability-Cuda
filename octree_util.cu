#include "octree_util.cu.h"

__global__ void fillOutKernel(Box box, float3 distance, Array<float3> input,
                              Array<float3> output) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < input.length; i += stride) {
        const auto in = input.elements[i];
        auto& out = output.elements[i];
        float3 delta = in - box.center;
        bool inside = isInBox(delta, box);
        if (inside) {
            // out = make_float3(222, 222, 222);
            out = distance;
            // atomicAdd(&(output.elements[i].x), distance.x);
            // atomicAdd(&(output.elements[i].x), 1.f);
            // atomicAdd(&(output.elements[i].z), 1.f);
        }
    }
}

