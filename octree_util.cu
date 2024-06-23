#include "cuda_runtime_api.h"
#include "octree_util.cu.h"
#include <cstdio>
#include <iostream>
#include <ostream>
#include <strings.h>
// #include <cstring>

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

template <typename Tin>
__device__ void copyHeap(Tin*& heapPtr, size_t size, Tin*& gpuPtr) {
    memcpy(gpuPtr, heapPtr, size);
}

__global__ void copyNode(Node* from, Node* to, size_t count) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < count; i += stride) {
        to[i] = from[i];
    }
}

__host__ void copyTreeOnCpuRecursion(Node& parent) {
    if (parent.leaf or parent.raw)
        return;
    // std::cout << &parent << std::endl;
    // Node* family = new Node[parent.childrenCount]();
    Node* family;
    // std::cout << family << std::endl;
    cudaMallocManaged(&family, parent.childrenCount * sizeof(Node));
    // std::cout << family << std::endl;
    CUDA_CHECK_ERROR("Managed alloc");
    Node* from = parent.childrenArr;
    Node* to = family;
    auto number = parent.childrenCount;
    copyNode<<<1, 32>>>(from, to, number);

    // std::cout << "f";
    // cudaMemcpy(family, parent.childrenArr, 1 * sizeof(Node),
    // cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Copy recurs");
    parent.childrenArr = family;
    for (uint c = 0; c < parent.childrenCount; c++) {
        Node& child = parent.childrenArr[c];
        copyTreeOnCpuRecursion(child);
    }
}

__host__ void copyTreeOnCpu(Node* gpuRoot, Node*& root) {
    // std::cout << root << std::endl;
    cudaMallocManaged(&root, sizeof(Node));
    cudaMemcpy(root, gpuRoot, 1 * sizeof(Node), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // std::cout << root->childrenArr << std::endl;
    CUDA_CHECK_ERROR("Copy root");
    copyTreeOnCpuRecursion(*root);
}

__host__ uint countLeaf(Node& node) {
    size_t count = 0;
    // std::cout << &node << std::endl;
    for (uint c = 0; c < node.childrenCount; c++) {
        // std::cout << node->childrenArr << std::endl;
        Node& child = node.childrenArr[c];
        // std::cout << &child << std::endl;
        if (child.leaf or child.raw)
            count++;
        else
            count += countLeaf(child);
    }
    return count;
}

__global__ void deleteTreeKernel(Node nodePTR) {
    auto& node = nodePTR;
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    // if (index == 0)
    // printf("\n%p", &nodePTR);
    for (auto i = index; i < node.childrenCount; i += stride) {
        Node child = node.childrenArr[i];
        if (not(child.raw or child.leaf))
            deleteTreeKernel<<<1, 32>>>(child);
    }
    __syncthreads();
    if (index == 0)
        cudaFree(node.childrenArr);
}
__global__ void deleteTreeKernel(Node* nodePTR) {
    auto& node = nodePTR[0];
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    // if (index == 0)
    // printf("\n%p", nodePTR);
    for (auto i = index; i < node.childrenCount; i += stride) {
        Node& child = node.childrenArr[i];
        if (not(child.raw or child.leaf))
            deleteTreeKernel<<<1, 32>>>(child);
    }
    __syncthreads();
    if (index == 0)
        cudaFree(node.childrenArr);
}
__host__ void deleteTree(Node* nodePTR) {
    cudaDeviceSynchronize();
    deleteTreeKernel<<<1, 32>>>(nodePTR);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("deletion");
}

float3* fill_recus(Node node, float3* ptr) {
    // std::printf("rec");
    // std::cout << std::endl;
    float3* movingPTR = ptr;
    for (uint i = 0; i < node.childrenCount; i++) {
        Node child = node.childrenArr[i];
        bool endpoint = not(child.leaf or child.raw or isNullBox(child.box));
        bool isValid =
            (not isNullBox(child.box)) and ((child.leaf or child.raw) and child.validity);
            // (not isNullBox(child.box)) and ((child.leaf ) and child.validity);
        if (endpoint)
            movingPTR = fill_recus(child, movingPTR);
        else if (isValid) {
            std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", child.box.center.x,
                        child.box.center.y, child.box.center.z);
            std::printf("\nsize x: %.1f y: %.1f z: %.1f ", child.box.topOffset.x,
                        child.box.topOffset.y, child.box.topOffset.z);
            std::cout << std::endl;
            movingPTR[0] = child.box.center;
            movingPTR++;
        }
    }
    return movingPTR;
}

__host__ Array<float3> extractValidAsArray(Node node) {
    auto count = countLeaf(node);
    Array<float3> output;
    output.length = count;
    output.elements = new float3[count * sizeof(float3)];
    auto endPtr = fill_recus(node, output.elements);
    uint validCount = endPtr - output.elements;
    printf("valid points: %d", validCount);
    std::cout << std::endl;
    float3* newPtr = new float3[validCount];
    // float3* newPtr;
    // memcpy(newPtr, output.elements, validCount * sizeof(float3));
    auto pin = newPtr;
    for (auto p = output.elements; p < endPtr; p++) {
        pin[0] = p[0];
        // std::printf("center x: %.1f y: %.1f z: %.1f ", p[0].x,
                    // p[0].y, p[0].z);
        // std::cout << std::endl;
        pin++;
    }
    delete[] output.elements;
    output.elements = newPtr;
    output.length = validCount;

    return output;
}
