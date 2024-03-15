#include "unified_math_cuda.cu.h"

inline __host__ __device__ float magnitude(float3 vec) {
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__host__ __device__ Quaternion quatFromVectAngle(float3 axis, float angle) {
    float sina_2, cosa_2;
    sincosf(angle / 2, &sina_2, &cosa_2);
    float mag = magnitude(axis);
    Quaternion result =
        make_float4(cosa_2, sina_2 * axis.x / mag, sina_2 * axis.y / mag,
                    sina_2 * axis.z / mag);
    // make_float4(sina_2 * axis.x, sina_2 * axis.y, sina_2 * axis.z, cosa_2);
    return result;
}
