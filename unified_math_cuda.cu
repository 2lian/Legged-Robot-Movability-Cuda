// #include "unified_math_cuda.cu.h"

inline __host__ __device__ float magnitude(float3 vec) {
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__host__ __device__ float3 qtRotate(Quaternion q, const float3 v) {
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

__host__ __device__ Quaternion qtInvert(Quaternion quat) {
    float qt2 =
        quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w;
    Quaternion result =
        make_float4(quat.x / qt2, -quat.y / qt2, -quat.z / qt2, -quat.w / qt2);
    return result;
}

__host__ __device__ float3 qtInvRotate(Quaternion q, const float3 v) {
    return qtRotate(qtInvert(q), v);
}

__host__ __device__ Quaternion qtMultiply(Quaternion q1, Quaternion q2) {
    float w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    float x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    float y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    float z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return make_float4(x, y, z, w);
}

__host__ __device__ Quaternion quatFromVectAngle(float3 axis, float angle) {
    float sina_2, cosa_2;
    sincosf(angle / 2, &sina_2, &cosa_2);
    float mag = magnitude(axis);
    Quaternion result =
        make_float4(sina_2, cosa_2 * axis.x / mag, cosa_2 * axis.y / mag,
                    cosa_2 * axis.z / mag);
    // make_float4(sina_2 * axis.x, sina_2 * axis.y, sina_2 * axis.z, cosa_2);
    return result;
}

__host__ __device__ float3 rpyFromQuat(const Quaternion q) {
    float3 rollPitchYaw;
    const auto x = q.x;
    const auto y = q.y;
    const auto z = q.z;
    const auto w = q.w;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (w * x + y * z);
    double cosr_cosp = 1 - 2 * (x * x + y * y);
    rollPitchYaw.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
        rollPitchYaw.y =
            std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        rollPitchYaw.y = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (w * z + x * y);
    double cosy_cosp = 1 - 2 * (y * y + z * z);
    rollPitchYaw.z = std::atan2(siny_cosp, cosy_cosp);
    return rollPitchYaw;
}
