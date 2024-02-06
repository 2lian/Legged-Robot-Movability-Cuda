#import "HeaderCUDA.h"
#import "one_leg.cu.h"

__device__ void rotateInPlace(float3 point, float z_rot, float& cos_memory,
                              float& sin_memory) {
    float cos;
    float sin;
    sincosf(z_rot, &sin, &cos);
    float buffer = point.x * sin;
    point.x = point.x * cos - point.y * sin;
    point.y = buffer + point.y * cos;
    return;
}
