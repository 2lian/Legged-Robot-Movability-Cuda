#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "octree_util.cu.h"
#include "one_leg.cu.h"
#include "settings.h"
#include "several_leg_octree.cu.h"
#include "unified_math_cuda.cu.h"
#include <cstdio>

// __device__ constexpr auto AngleSample_D = AngleSample;
// __device__ constexpr auto AngleMinMax_D = AngleMinMax;
// __device__ constexpr auto LegMount_D = LegMount;
//
typedef struct Node {
    Box box;
    bool validity;
    bool leaf;
    bool raw;
    uchar childrenCount = MaxChildQuad;
    Node* childrenArr;
} Node;

__global__ void validity_child(Node parent, const Array<float3> input,
                               const LegDimensions leg) {

    auto box = parent.box;
    auto validity = parent.validity;
    __shared__ bool childrens[MaxChildQuad];
    __shared__ bool validLeafs[MaxChildQuad];
    __shared__ bool globalValidity[MaxChildQuad];
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    for (auto t = threadIdx.x; t < MaxChildQuad; t++) {
        childrens[t] = false;
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
        auto footholdIndex = threadQuadranId / totalFootholdSample;
        auto angleIndex = threadQuadranId % totalFootholdSample;
        Node& node = parent.childrenArr[childIndex];
        if (node.validity) { // DEADQUADRAN or already processed
            // std::printf("already done");
            continue;
        }
        bool tooSmall = node.leaf;
        Box new_box = node.box;
        auto& thisChildrenShouldSpawn = childrens[childIndex];
        auto& thisGlobalValidity = globalValidity[childIndex];
        auto& thisValidLeaf = validLeafs[childIndex];

        float3 distToNewBox;

        auto body = new_box.center;
        bool reachabilityEdgeInTheBox = false;
        bool reachability = validity;
        // float3 farthestDist = make_float3(0, 0, 0);

        const auto foothold = input.elements[footholdIndex];

        const Quaternion quat = QuaternionFromAngleIndex(angleIndex);
        uchar reachabilityCount = 0;
        uchar crossBoxCount = 0;

        for (uchar legN = 0; legN < LegCount; legN++) {
            auto vect = foothold - body;
            auto thisleg = leg;
            thisleg.body_angle = LegMount_D[legN];
            bool subReachability = distance(vect, leg, quat);
            bool subCrossBox = linorm(vect) < linorm(new_box.topOffset);
            // farthestDist = (linorm(farthestDist) > linorm(vect)) ? vect : farthestDist;
            crossBoxCount += subCrossBox;
            reachabilityCount += subReachability;
        }
        if (crossBoxCount > 0)
            reachabilityEdgeInTheBox = true;
        if (reachabilityCount >= LegNumberForStab)
            reachability = true;

        bool validLeaf = reachability and not reachabilityEdgeInTheBox;
        bool childrenShouldSpawn =
            reachabilityEdgeInTheBox and (not tooSmall) and (not validLeaf);
        if (childrenShouldSpawn)
            thisChildrenShouldSpawn = true;
        if (reachability)
            thisGlobalValidity = true;
        if (validLeaf)
            thisValidLeaf = true;
    }
    __syncthreads();

    for (auto t = threadIdx.x; t < parent.childrenCount; t++) {
        auto& node = parent.childrenArr[t];
        auto& isChildrenShouldSpawn = childrens[t];
        auto& isGlobalValid = globalValidity[t];
        auto& isValidLeaf = validLeafs[t];
        if (isGlobalValid)
            node.validity = true;
        if (isValidLeaf)
            node.leaf = true;
        if (not isChildrenShouldSpawn)
            node.leaf = true;
    }
}

__global__ void branchKernel(Node* parentPTR, Array<float3> input,
                             const LegDimensions leg, uchar depth) {
    Node& parent = *parentPTR;
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    auto maxComputeIndex = parent.childrenCount;
    uchar quadCount = 3;
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
    if (not goDeeperDEEPERRRR) {
        if (index == 0) {
            std::printf("\nComputing depth %d", depth);
            std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", parent.box.center.x,
                        parent.box.center.y, parent.box.center.z);
            std::printf("\nleaf: %d", parent.leaf);
            std::printf("\nraw: %d", parent.raw);
            std::printf("\nvalidity: %d", parent.validity);
            cudaMalloc(&parent.childrenArr, sizeof(Node) * parent.childrenCount);
            // parent.childrenArr = (Node*)malloc(sizeof(Node) * MaxChildQuad);
        }
    } else {
        if (index == 0) {
            std::printf("\nForking depth %d", depth);
            std::printf("\ncenter x: %.1f y: %.1f z: %.1f ", parent.box.center.x,
                        parent.box.center.y, parent.box.center.z);
            std::printf("\nleaf: %d", parent.leaf);
            std::printf("\nraw: %d", parent.raw);
            std::printf("\nvalidity: %d", parent.validity);
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
            bool branch = not node.leaf;
            if (branch) {
                constexpr uint maxBlockSize = 1024 / 4;
                int blockSize = min(node.childrenCount, maxBlockSize);
                int numBlock = (node.childrenCount + blockSize - 1) / blockSize;
                branchKernel<<<numBlock, blockSize>>>(&node, input, leg, depth + 1);
            } else {
                std::printf("\nleaf x: %.1f y: %.1f z: %.1f ", node.box.center.x,
                node.box.center.y, node.box.center.z);
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
            node.box = new_box;
            continue;
        }
        if (subQuadCount <= 0) {
            // no new quadran possible so we mark it as leaf, put the box in, needs
            // processing but no malloc
            node.leaf = true;
            node.raw = false;
            node.validity = false;
            node.box = new_box;
        } else {
            // new quadran so not a leaf and needs processing
            node.leaf = false;
            node.raw = true;
            node.validity = false;
            node.box = new_box;
        }
    }
    if (not goDeeperDEEPERRRR) {
        if (index == 0) {
            std::printf("\nComputing!\n");
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
            std::printf("<%d %d>", numBlock, blockSize);
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

float apply_oct(Array<float3> input, LegDimensions dim) {
    Array<float3> gpu_in{};
    gpu_in.length = input.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(float3));
    CUDA_CHECK_ERROR("cudaMalloc gpu_in.elements");

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(float3),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("cudaMemcpy gpu_in.elements");

    constexpr int blockSize = 64;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    Box box;
    box.center = make_float3(BoxCenter[0], BoxCenter[1], BoxCenter[2]);
    box.topOffset = make_float3(BoxSize[0], BoxSize[1], BoxSize[2]);
    Node* deviceRoot;
    Node root;
    // cudaMallocManaged(&managedRoot, sizeof(Node));
    root.box = box;
    root.raw = true;
    root.leaf = false;
    root.validity = false;
    root.childrenCount = MaxChildQuad;
    cudaMalloc(&deviceRoot, sizeof(Node));
    cudaMemcpy(deviceRoot, &root, sizeof(Node), cudaMemcpyHostToDevice);
    // managedRoot[0] = root;
    const uint max_quad_ind = pow(8, 2);
    numBlock = (max_quad_ind + blockSize - 1) / blockSize;
    // recursive_kernel<<<1, 24>>>(box, gpu_in, dim, gpu_out, 30);
    // Prepare
    cudaEvent_t start, stop;
    cudaStream_t stream;
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, stream);
    // Do something on GPU
    for (uint d = 0; d < MAX_DEPTH; d++) {
        branchKernel<<<numBlock, blockSize, 0, stream>>>(deviceRoot, gpu_in, dim, 0);
        cudaStreamSynchronize(stream);
    }
    // branchKernel<<<numBlock, blockSize, 0, stream>>>(managedRoot, gpu_in, dim, 0);
    // cudaStreamSynchronize(stream);
    // branchKernel<<<numBlock, blockSize, 0, stream>>>(managedRoot, gpu_in, dim, 0);
    // cudaStreamSynchronize(stream);
    // branchKernel<<<numBlock, blockSize, 0, stream>>>(managedRoot, gpu_in, dim, 0);
    // recursive_kernel<<<numBlock, blockSize>>>(box, gpu_in, dim, gpu_out, 0, 0, false);
    // recursive_kernel<<<1, 24>>>(box, gpu_in, dim, gpu_out, 0, 0);
    // Stop event and sync
    // cudaDeviceSynchronize();
    // std::this_thread::sleep_for(1s);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK_ERROR("Kernel launch");

    cudaDeviceSynchronize();

    cudaFree(gpu_in.elements);
    return elapsedTime;
}
