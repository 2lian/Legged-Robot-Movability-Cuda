#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_runtime_api.h"
#include "octree_util.cu.h"
#include "one_leg.cu.h"
#include "settings.h"
#include "unified_math_cuda.cu.h"
#include <__clang_cuda_builtin_vars.h>

typedef struct Node {
    Box box;
    bool validity;
    bool leaf;
    bool raw;
    const uchar childrenCount = MaxChildQuad;
    Node* childrenArr;
} Node;

__global__ void validity_octree(Node parent, const Array<float3> input,
                                const LegDimensions leg, PlaneImage output, uchar depth) {

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

    uchar quadCount = 3;
    bool small[3];
    small[0] = abs(box.topOffset.x) < MIN_BOX_X;
    small[1] = abs(box.topOffset.y) < MIN_BOX_Y;
    small[2] = abs(box.topOffset.z) < MIN_BOX_Z;
    if (small[0])
        quadCount -= 1;
    if (small[1])
        quadCount -= 1;
    if (small[2])
        quadCount -= 1;

    // quad_count = 3;
    const uint max_quad_ind = pow(pow(2, SUB_QUAD), quadCount);
    const size_t totalFootholdSample = input.length;
    constexpr size_t totalAngleSample = AngleSample[0] * AngleSample[1] * AngleSample[2];
    const size_t maxComputeIndex = max_quad_ind * totalFootholdSample * totalAngleSample;

    for (size_t computeIndex = index; computeIndex < maxComputeIndex;
         computeIndex += stride) {
        auto quadranIndex = computeIndex / (totalAngleSample * totalFootholdSample);
        auto threadQuadranId = computeIndex % (totalAngleSample * totalFootholdSample);
        auto footholdIndex = threadQuadranId / totalFootholdSample;
        auto angleIndex = threadQuadranId % totalFootholdSample;
        Box new_box;
        uchar missingQuad;

        uint orderedQuadIndex =
            CreateChildBox(box, new_box, quadCount, quadranIndex, small, missingQuad);
        if (threadQuadranId == 0) {
            auto& node = parent.childrenArr[quadranIndex];
            node.box = new_box;
        }
        auto& thisChildrenShouldSpawn = childrens[orderedQuadIndex];
        auto& thisGlobalValidity = globalValidity[orderedQuadIndex];
        auto& thisValidLeaf = validLeafs[orderedQuadIndex];

        if (missingQuad == DEADQUADRAN or thisChildrenShouldSpawn)
            continue;

        auto subQuadCount = 3 - missingQuad;
        bool tooSmall = subQuadCount <= 0;
        float3 distToNewBox;

        auto body = new_box.center;
        // distance.y = 0.1;
        bool reachabilityEdgeInTheBox = false;
        bool reachability = validity;
        float3 farthestDist = make_float3(0, 0, 0);

        const auto foothold = input.elements[footholdIndex];

        Quaternion quat = QuaternionFromAngleIndex(angleIndex);
        uchar reachabilityCount = 0;
        uchar crossBoxCount = 0;

        for (uchar legN = 0; legN < LegCount; legN++) {
            auto vect = foothold - body;
            auto thisleg = leg;
            thisleg.body_angle = LegMount[legN];
            bool subReachability = distance(vect, leg, quat);
            bool subCrossBox = linorm(vect) < linorm(new_box.topOffset);
            farthestDist = (linorm(farthestDist) > linorm(vect)) ? vect : farthestDist;
            crossBoxCount += subCrossBox;
            reachabilityCount += subReachability;
        }
        if (crossBoxCount > 0)
            reachabilityEdgeInTheBox = true;
        if (reachabilityCount >= LegNumberForStab)
            reachability = true;

        bool validLeaf = reachability and not reachabilityEdgeInTheBox;
        bool childrenShouldSpawn = reachabilityEdgeInTheBox and (not tooSmall) and
                                   (depth < MAX_DEPTH) and (not validLeaf);
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
        if (isValidLeaf and not isChildrenShouldSpawn)
            node.leaf = true;
    }
}

__global__ void branchKernel(const Node parent) {

    // cudaMalloc(&parent.childrenArr, sizeof(Node) * parent.childrenCount);
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    // auto maxComputeIndex = parent.childrenCount;
    uchar quadCount = 3;
    bool small[3];
    small[0] = abs(parent.box.topOffset.x) < MIN_BOX_X;
    small[1] = abs(parent.box.topOffset.y) < MIN_BOX_Y;
    small[2] = abs(parent.box.topOffset.z) < MIN_BOX_Z;
    if (small[0])
        quadCount -= 1;
    if (small[1])
        quadCount -= 1;
    if (small[2])
        quadCount -= 1;
    auto maxComputeIndex = pow(pow(2, SUB_QUAD), quadCount);
    for (size_t computeIndex = index; computeIndex < maxComputeIndex;
         computeIndex += stride) {
        Node& node = parent.childrenArr[computeIndex];

        bool end = node.leaf;
        if (end) {
            // please fill image on other launch
            continue;
        }

        bool deeperDEEPER = not node.raw;
        if (deeperDEEPER) {
            constexpr uint maxBlockSize = 1024 / 4;
            int blockSize = min(node.childrenCount, maxBlockSize);
            int numBlock = (node.childrenCount + blockSize - 1) / blockSize;
            branchKernel<<<numBlock, blockSize>>>(node);
            continue;
        }

        // here we are left with uninitialized nodes
        auto quadranIndex = computeIndex;
        Box new_box;
        uchar missingQuad;

        // constexpr bool small[3] = {false, false, false};
        uint orderedQuadIndex = CreateChildBox(parent.box, new_box, 3, (uint)quadranIndex,
                                               small, missingQuad);
        node.childrenCount = (3-missingQuad);
        cudaMalloc(&node.childrenArr, sizeof(Node) * parent.childrenCount);
        __syncthreads();
    }
}
