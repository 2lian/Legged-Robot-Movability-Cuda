#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "octree_util.cu.h"
#include "one_leg.cu.h"
#include "settings.h"
#include "several_leg_octree.cu.h"
#include "unified_math_cuda.cu.h"
#include <cstdio>
#include <iostream>
#include <ostream>

// __device__ constexpr auto AngleSample_D = AngleSample;
// __device__ constexpr auto AngleMinMax_D = AngleMinMax;
// __device__ constexpr auto LegMount_D = LegMount;
//
__global__ void validity_child(Node parent, const Array<float3> input,
                               const LegDimensions leg) {

    auto box = parent.box;
    auto validity = parent.validity;
    __shared__ bool onEdge[MaxChildQuad];
    __shared__ bool validLeafs[MaxChildQuad];
    __shared__ bool globalValidity[MaxChildQuad];
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    // if (index == 1)
    // printf("her");
    for (auto t = threadIdx.x; t < MaxChildQuad; t++) {
        onEdge[t] = false;
        validLeafs[t] = false;
        globalValidity[t] = false;
    }

    // quad_count = 3;
    const uint max_child_ind = parent.childrenCount;
    const size_t totalFootholdSample = input.length;
    constexpr size_t totalAngleSample =
        AngleSample_D[0] * AngleSample_D[1] * AngleSample_D[2];
    const size_t maxComputeIndex = max_child_ind * totalFootholdSample * totalAngleSample;
    // std::printf("hey");

    for (size_t computeIndex = index; computeIndex < maxComputeIndex;
         computeIndex += stride) {
        auto childIndex = computeIndex / (totalAngleSample * totalFootholdSample);
        auto threadQuadranId = computeIndex % (totalAngleSample * totalFootholdSample);
        auto footholdIndex = threadQuadranId / totalAngleSample;
        auto angleIndex = threadQuadranId % totalAngleSample;
        bool rotationActive = parent.box.topOffset.x < EnableRotBelow;
        if (not rotationActive and angleIndex > 0) {
            continue;
        }
        float margin = (rotationActive) ? 0 : EnableRotBelow / 3;
        Node& node = parent.childrenArr[childIndex];
        if (node.validity) { // DEADQUADRAN or already processed
            // std::printf("already done");
            continue;
        }
        // bool tooSmall = node.leaf;
        Box new_box = node.box;
        auto& thisOnEdge = onEdge[childIndex];
        auto& thisGlobalValidity = globalValidity[childIndex];
        auto& thisValidLeaf = validLeafs[childIndex];

        // float3 distToNewBox;

        auto body = new_box.center;
        bool reachabilityEdgeInTheBox = false;
        bool reachability = validity;
        // float3 farthestDist = make_float3(0, 0, 0);

        const auto foothold = input.elements[footholdIndex];
        auto vect = foothold - body;
        Box elongateBox = box;
        elongateBox.center = make_float3(0, 0, 0);
        elongateBox.topOffset =
            elongateBox.topOffset +
            (leg.body + leg.coxa_length + leg.femur_length + leg.tibia_length);
        if (not isInBox(vect, elongateBox))
            continue;

        const Quaternion quat = QuaternionFromAngleIndex(angleIndex);
        uchar reachabilityCount = 0;
        uchar crossBoxCount = 0;

        // printf("q: %d, f: %d , x: %f , y: %f , z: %f\n", (int)childIndex,
        // (int)footholdIndex, foothold.x, foothold.y, foothold.z);
        // printf("p");
        for (uchar legN = 0; legN < LegCount; legN++) {
            auto v = vect;
            auto thisleg = leg;
            thisleg.body_angle = LegMount_D[legN];
            bool subReachability = distance(v, thisleg, quat);
            float distEdgeRaw = linormRaw(new_box.topOffset);
            bool subCrossBox;
            constexpr float conradSqrd = convexRadius * convexRadius;
            if (distEdgeRaw > conradSqrd) {
                Box zerobox = new_box;
                zerobox.center = make_float3(0, 0, 0);
                zerobox.topOffset = zerobox.topOffset + margin;
                subCrossBox = isInBox(v, new_box);
            } else
                subCrossBox = linormRaw(v) < linormRaw(new_box.topOffset) + margin;

            // farthestDist = (linorm(farthestDist) > linorm(v)) ? v : farthestDist;
            crossBoxCount += subCrossBox;
            reachabilityCount += subReachability;
        }
        if (crossBoxCount > LegCount - LegNumberForStab)
            reachabilityEdgeInTheBox = true;
        if (reachabilityCount >= LegNumberForStab)
            reachability = true;

        bool validLeaf = reachability and not reachabilityEdgeInTheBox;
        // bool childrenShouldSpawn = reachabilityEdgeInTheBox and (not tooSmall);
        bool childrenOnEdge = reachabilityEdgeInTheBox;
        if (childrenOnEdge) {
            // printf("o");
            thisOnEdge = true;
        }
        if (reachability) {
            // std::printf("r");
            thisGlobalValidity = true;
        }
        if (validLeaf) {
            // std::printf("l");
            thisValidLeaf = true;
        }
    }
    __syncthreads();

    for (auto t = threadIdx.x; t < parent.childrenCount; t++) {
        auto& node = parent.childrenArr[t];
        auto& isOnEdge = onEdge[t];
        auto& isGlobalValid = globalValidity[t];
        auto& isValidLeaf = validLeafs[t];
        if (isGlobalValid) {
            node.validity = true;
        }
        if (isValidLeaf) {
            // std::printf("found");
            node.leaf = true;
        }
        if (isOnEdge and not isValidLeaf) {
            // std::printf("far");
            node.onEdge = true;
        }
    }
}

__host__ void branchCpu(Node& parent, Array<float3> input, const LegDimensions leg,
                        uchar depth, cudaStream_t stream) {
    // const bool goDeeperDEEPERRRR = not parent.raw;
    // std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", parent.box.center.x,
    // parent.box.center.y, parent.box.center.z);
    // std::printf("\nleaf: %d", parent.leaf);
    // std::printf("\nraw: %d", parent.raw);
    // std::printf("\nvalidity: %d", parent.validity);
    if (parent.raw) {
        // here we are left with uninitialized raw nodes
        cudaMallocManaged(&parent.childrenArr, parent.childrenCount * sizeof(Node));
        for (uint c = 0; c < parent.childrenCount; c++) { // inits all subbox
            Node& node = parent.childrenArr[c];
            auto quadranIndex = c;
            Box new_box;
            uchar missingQuad;

            // we need to find the box of the uninitialized
            bool small[3] = {0, 0, 0};
            uint orderedQuadIndex = CreateChildBox(
                parent.box, new_box, 3, (uint)quadranIndex, small, missingQuad);
            auto subQuadCount = (3 - missingQuad);
            node.childrenCount = MaxChildQuad;
            if (missingQuad == DEADQUADRAN) {
                // std::printf("DEAD %d ", index);
                node.leaf = true;
                node.raw = false;
                node.validity = true;
                node.onEdge = true;
                node.box = NullBox;
                continue;
            }
            node.onEdge = false;
            node.validity = false;
            if (subQuadCount <= 0) {
                // no new quadran possible so we mark it as leaf, put the box in, needs
                // processing but no malloc
                node.leaf = true;
                node.raw = false;
                node.box = new_box;
            } else {
                // new quadran so not a leaf and needs processing
                node.leaf = false;
                node.raw = true;
                node.box = new_box;
            }
            // parent.childrenArr[c] = node; // write to managed mem
        }

        // std::cout << "\nComputing!\n" << std::endl;
        parent.raw = false;
        // return;
        const uint max_child_ind = parent.childrenCount;
        const size_t totalFootholdSample = input.length;
        constexpr size_t totalAngleSample =
            AngleSample[0] * AngleSample[1] * AngleSample[2];
        const size_t maxComputeIndex =
            max_child_ind * totalFootholdSample * totalAngleSample;

        constexpr uint maxBlockSize = 1024 / 4;

        int blockSize = min(maxComputeIndex, (typeof(maxComputeIndex))maxBlockSize);
        int numBlock = (maxComputeIndex + blockSize - 1) / blockSize;
        // std::printf("<%d %d>", numBlock, blockSize);
        std::cout << &parent << std::endl;
        // cudaDeviceSynchronize();
        // auto p = parent;
        validity_child<<<numBlock, blockSize, 0, stream>>>(parent, input, leg);
        // std::cout << "\nComputing!\n" << std::endl;
        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("vailidy cpu");

    } else {
        // std::cout << &parent << std::endl;
        for (uint c = 0; c < parent.childrenCount; c++) {
            Node& node = parent.childrenArr[c];
            // std::cout << &node << std::endl;
            if (not node.onEdge)
                node.leaf = true;
            bool branch = not node.leaf;
            if (branch) {
                branchCpu(node, input, leg, depth + 1, stream);
            }
        }
    }
}

__global__ void branchKernel(Node* parentPTR, Array<float3> input,
                             const LegDimensions leg, uchar depth) {
    Node& parent = *parentPTR;
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    auto maxComputeIndex = parent.childrenCount;
    // uchar quadCount = 3;
    // bool small[3];
    // small[0] = abs(parent.box.topOffset.x) < MIN_BOX_X;
    // small[1] = abs(parent.box.topOffset.y) < MIN_BOX_Y;
    // small[2] = abs(parent.box.topOffset.z) < MIN_BOX_Z;
    // if (small[0])
    //     quadCount -= 1;
    // if (small[1])
    //     quadCount -= 1;
    // if (small[2])
    //     quadCount -= 1;
    bool small[3] = {0, 0, 0};
    // auto maxComputeIndex = pow(pow(2, SUB_QUAD), quadCount);
    const bool goDeeperDEEPERRRR = not parent.raw;
    if (index == 0 and not goDeeperDEEPERRRR) {
        std::printf("\n\nDepth %d", depth);
        std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", parent.box.center.x,
                    parent.box.center.y, parent.box.center.z);
        std::printf("\nsize x: %.1f y: %.1f z: %.1f ", parent.box.topOffset.x,
                    parent.box.topOffset.y, parent.box.topOffset.z);
    }
    if (not goDeeperDEEPERRRR) {
        if (index == 0) {
            // std::printf("\n\nComputing depth %d", depth);
            // std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", parent.box.center.x,
            // parent.box.center.y, parent.box.center.z);
            // std::printf("\nleaf: %d", parent.leaf);
            // std::printf("\nraw: %d", parent.raw);
            // std::printf("\nvalidity: %d", parent.validity);
            cudaMalloc(&parent.childrenArr, sizeof(Node) * parent.childrenCount);
            // parent.childrenArr = (Node*)malloc(sizeof(Node) * MaxChildQuad);
        }
    } else {
        if (index == 0) {
            // std::printf("\n\nForking depth %d", depth);
            // std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", parent.box.center.x,
            // parent.box.center.y, parent.box.center.z);
            // std::printf("\nleaf: %d", parent.leaf);
            // std::printf("\nraw: %d", parent.raw);
            // std::printf("\nvalidity: %d", parent.validity);
        }
    }
    __syncthreads();
    __threadfence_block();
    // void __threadfence_system();

    for (size_t computeIndex = index; computeIndex < maxComputeIndex;
         computeIndex += stride) {

        if (goDeeperDEEPERRRR) {
            // std::printf("[%d] DEEEEEEEPER level: %d |", index, depth);
            Node& node = parent.childrenArr[computeIndex];
            if (not node.onEdge)
                node.leaf = true;
            bool branch = not node.leaf;
            if (branch) {
                constexpr uint maxBlockSize = 1024 / 4;
                int blockSize = min(node.childrenCount, maxBlockSize);
                int numBlock = (node.childrenCount + blockSize - 1) / blockSize;
                branchKernel<<<numBlock, blockSize>>>(&node, input, leg, depth + 1);
            } else {
                // std::printf("\nleaf x: %.1f y: %.1f z: %.1f validity: %d raw: %d",
                // node.box.center.x, node.box.center.y, node.box.center.z,
                // node.validity, node.raw);
            }
            continue; // goto next computeIndex
        }

        Node& node = parent.childrenArr[computeIndex];
        // return;

        // here we are left with uninitialized raw nodes
        auto quadranIndex = computeIndex;
        Box new_box;
        uchar missingQuad;

        // we need to find the box of the uninitialized
        uint orderedQuadIndex = CreateChildBox(parent.box, new_box, 3, (uint)quadranIndex,
                                               small, missingQuad);
        auto subQuadCount = (3 - missingQuad);
        node.childrenCount = MaxChildQuad;

        // if quadran is dead, we mark it as leaf, valid, with a NullBox and do not
        // cudaMalloc
        if (missingQuad == DEADQUADRAN) {
            // std::printf("DEAD %d ", index);
            node.leaf = true;
            node.raw = false;
            node.validity = true;
            node.onEdge = true;
            node.box = NullBox;
            continue;
        }
        node.onEdge = false;
        node.validity = false;
        if (subQuadCount <= 0) {
            // no new quadran possible so we mark it as leaf, put the box in, needs
            // processing but no malloc
            node.leaf = true;
            node.raw = false;
            node.box = new_box;
        } else {
            // new quadran so not a leaf and needs processing
            node.leaf = false;
            node.raw = true;
            node.box = new_box;
        }
    }
    if (not goDeeperDEEPERRRR) {
        if (index == 0) {
            // std::printf("\nComputing!\n");
            parent.raw = false;
            // return;
            const uint max_child_ind = parent.childrenCount;
            const size_t totalFootholdSample = input.length;
            constexpr size_t totalAngleSample =
                AngleSample_D[0] * AngleSample_D[1] * AngleSample_D[2];
            const size_t maxComputeIndex =
                max_child_ind * totalFootholdSample * totalAngleSample;

            constexpr uint maxBlockSize = 1024 / 4;

            int blockSize = min(maxComputeIndex, (typeof(maxComputeIndex))maxBlockSize);
            int numBlock = (maxComputeIndex + blockSize - 1) / blockSize;
            // std::printf("<%d %d>", numBlock, blockSize);
            validity_child<<<numBlock, blockSize>>>(parent, input, leg);
            // input.length = 10;
            // validity_child<<<1, 16>>>(parent, input, leg);
        }
    }
}

#define CUDA_CHECK_ERROR(errorMessage)                                                   \
    do {                                                                                 \
        cudaError_t err = cudaGetLastError();                                            \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error in %s: %s\n", errorMessage,                      \
                    cudaGetErrorString(err));                                            \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// __constant__ float3 cst[20000];

__host__ float apply_oct(Array<float3> input, LegDimensions dim, Array<float3>& output) {
    Array<float3> gpu_in{};
    Array<float3> gpu_out{};
    gpu_in.length = input.length;
    gpu_out.length = output.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(float3));
    // cudaMalloc(cst, gpu_in.length * sizeof(float3));
    cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(float3));
    CUDA_CHECK_ERROR("cudaMalloc gpu_in.elements");

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(float3),
               cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(cst, input.elements, gpu_in.length * sizeof(float3),
    // cudaMemcpyHostToDevice);
    // gpu_in.elements = cst;
    CUDA_CHECK_ERROR("cudaMemcpy gpu_in.elements");

    constexpr int blockSize = 64;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    Box box;
    box.center = make_float3(BoxCenter[0], BoxCenter[1], BoxCenter[2]);
    box.topOffset = make_float3(BoxSize[0], BoxSize[1], BoxSize[2]);
    Node* deviceRoot;
    Node root;
    Node* managedRoot;
    cudaMallocManaged(&managedRoot, sizeof(Node));
    root.box = box;
    root.raw = true;
    root.leaf = false;
    root.validity = false;
    root.childrenCount = MaxChildQuad;
    managedRoot->box = box;
    managedRoot->raw = true;
    managedRoot->leaf = false;
    managedRoot->validity = false;
    managedRoot->childrenCount = MaxChildQuad;
    cudaMalloc(&deviceRoot, sizeof(Node));
    cudaMemcpy(deviceRoot, &root, sizeof(Node), cudaMemcpyHostToDevice);
    // managedRoot[0] = root;
    const uint max_quad_ind = pow(8, 2);
    numBlock = (max_quad_ind + blockSize - 1) / blockSize;
    // recursive_kernel<<<1, 24>>>(box, gpu_in, dim, gpu_out, 30);
    // Prepare
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 200000000);
    // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 100);
    CUDA_CHECK_ERROR("limit set");
    // ((size_t)MaxChildQuad * (size_t)(MAX_DEPTH)));
    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, stream);
    // Do something on GPU
    for (uint d = 0; d < MAX_DEPTH; d++) {
        branchKernel<<<numBlock, blockSize, 0, stream>>>(deviceRoot, gpu_in, dim, 0);
        // branchCpu(managedRoot[0], gpu_in, dim, 0, stream);
        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaStreamSynchronize(stream);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK_ERROR("Kernel launch");
    std::cout << "\n\ncompute done, building output" << std::endl;
    Node* rootPtr;
    copyTreeOnCpu(deviceRoot, rootPtr);
    std::cout << "deleting GPU mem" << std::endl;
    deleteTree(deviceRoot);
    std::cout << "counting leafs" << std::endl;
    std::cout << "total: " << countLeaf(*rootPtr) << std::endl;
    delete[] output.elements;
    auto newoutput = extractValidAsArray(*rootPtr);
    output.length = newoutput.length;
    output.elements = newoutput.elements;

    // std::cout << "total: " << countLeaf(*managedRoot) << std::endl;

    // makeImgFromTree<<<numBlock, blockSize, 0, stream>>>(deviceRoot, gpu_in,
    // gpu_out, 0);
    CUDA_CHECK_ERROR("Kernel img");

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
    // std::cout << "image done" << std::endl;

    cudaFree(gpu_in.elements);
    cudaFree(gpu_out.elements);
    return elapsedTime;
}
