#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"

// extern float sin(float x); // shuts up clangd
// extern float cos(float x);
// extern float sqrt(float x);

#define LOWER_SIDE 0
#define UPPER_SIDE 1

template <uchar up_or_down = LOWER_SIDE>
__forceinline__ __host__ __device__ float
min_femur_to_gripper_dist(const LegDimensions leg) {
    float tibia_angle;
    if constexpr (up_or_down == LOWER_SIDE) {
        tibia_angle = leg.min_angle_tibia;
    } else {
        tibia_angle = leg.max_angle_tibia;
    }
    const float x = leg.femur_length + leg.tibia_length * cos(tibia_angle);
    const float y = leg.tibia_length * sin(tibia_angle);

    return sqrt(x * x + y * y);
}

__forceinline__ __host__ __device__ float
max_femur_to_gripper_dist(const LegDimensions leg) {
    return leg.tibia_length + leg.femur_length;
}

template <uchar up_or_down>
__forceinline__ __host__ __device__ void saturated_femur(const LegDimensions leg,
                                                         float& x_out, float& y_out) {
    float femur_angle;
    if constexpr (up_or_down == LOWER_SIDE) {
        femur_angle = leg.min_angle_femur;
    } else {
        femur_angle = leg.max_angle_femur;
    }
    x_out = cos(femur_angle) * leg.femur_length;
    y_out = sin(femur_angle) * leg.femur_length;
}
