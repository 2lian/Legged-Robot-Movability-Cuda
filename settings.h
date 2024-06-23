#pragma once
#include "HeaderCUDA.h"
// #include "cuda_runtime_api.h"
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

#define CIRCLE_MARGIN 0.001 // margin in mm for inside/outside circles
// then points (circles of radius < CIRCLE_MARGIN)
#define REACH_USECASE 0         // alias for the reachability computation
#define DIST_USECASE 1          // alias for the distance computation
#define CIRCLE_ARR_ORDERED true // signifies that the circle array first holds circles

#define MAX_DEPTH 9
#define PI 3.14159265358979323846264338327950288419716939937510582097f
constexpr float MINBOXSIZE = 100;
constexpr unsigned int SUB_QUAD = 2;
constexpr uint MaxChildQuad = 1 << (SUB_QUAD * 3);
constexpr bool OutputOctree = true; // save octree result in image output
                                    // because this is not optimised, it is really slow

constexpr float BoxCenter[3] = {0, 0, 0};
// constexpr float BoxSize[3] = {4, 0.1, 4};
constexpr float BoxSize[3] = {50000, 50000, 50000};
constexpr float MIN_BOX[3] = {MINBOXSIZE, MINBOXSIZE, MINBOXSIZE};
constexpr float MIN_BOX_X = MINBOXSIZE;
constexpr float MIN_BOX_Y = MINBOXSIZE;
constexpr float MIN_BOX_Z = MINBOXSIZE;
constexpr uchar DEADQUADRAN = 1 << 7;

constexpr float EnableRotBelow = 50;
constexpr float convexRadius = 100;
constexpr uchar AngleSample[3] = {3, 3, 3};
__device__ constexpr uchar AngleSample_D[3] = {
    AngleSample[0], AngleSample[1], AngleSample[2]};
constexpr float AngleMinMax[6] = {-PI / 4, PI / 4, -PI / 8, PI / 8, -PI / 8, PI / 8};
__device__ constexpr float AngleMinMax_D[6] = {-PI / 4, PI / 4,  -PI / 8,
                                                            PI / 8,  -PI / 8, PI / 8};
constexpr uchar LegCount = 4;
constexpr float LegMount[LegCount] = {PI / 4 * 0, PI / 4 * 1, PI / 4 * 2, PI / 4 * 3};
__device__ constexpr float LegMount_D[LegCount] = {PI / 4 * 0, PI / 4 * 1,
                                                                PI / 4 * 2, PI / 4 * 3};
// constexpr uchar LegNumberForStab = LegCount;
constexpr uchar LegNumberForStab = 4;

constexpr Box NullBox = {{0, 0, 0}, {0, 0, 0}};
