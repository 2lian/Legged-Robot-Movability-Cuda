#pragma once
// #include "cuda_runtime_api.h"
#include <driver_types.h>

typedef float4 Quaternion;

__host__ __device__ inline float3 qtRotate(Quaternion q, const float3 v) {
    float t2 = q.x * q.y;
    float t3 = q.x * q.z;
    float t4 = q.x * q.w;
    float t5 = -q.y * q.y;
    float t6 = q.y * q.z;
    float t7 = q.y * q.w;
    float t8 = -q.z * q.z;
    float t9 = q.z * q.w;
    float t10 = -q.w * q.w;
    return make_float3(
        2.0f * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
        2.0f * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
        2.0f * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z);
}

__host__ __device__ inline Quaternion qtInvert(Quaternion quat) {
    float qt2 =
        quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w;
    Quaternion result =
        make_float4(quat.x / qt2, -quat.y / qt2, -quat.z / qt2, -quat.w / qt2);
    return result;
}

__host__ __device__ inline float3 qtInvRotate(Quaternion q, const float3 v) {
    return qtRotate(qtInvert(q), v);
}

__host__ __device__ Quaternion quatFromVectAngle(float3 axis, float angle);

struct QuaternionFunctor {
    Quaternion quat;

    __host__ __device__ QuaternionFunctor(Quaternion _quat) : quat(_quat) {}

    __device__ float3 operator()(float3 point) const {
        return qtRotate(quat, point);
        ;
    }
};

struct UnQuaternionFunctor {
    Quaternion quat;

    __host__ __device__ UnQuaternionFunctor(Quaternion _quat) : quat(_quat) {}

    __device__ float3 operator()(float3 point) const {
        return qtInvRotate(quat, point);
        ;
    }
};
