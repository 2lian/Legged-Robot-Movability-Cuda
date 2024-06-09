#pragma once
#include <tuple>

typedef unsigned char uchar;
constexpr float pI = 3.14159265358979323846264338327950288419716939937510582097f;

typedef struct Circle {
  public:
    float x;
    float y;
    float radius;
    bool attractivity;
} Cricle;

typedef Circle Intersect;

typedef struct LegDimensions {
  public:
    float body_angle;
    float body;
    float coxa_pitch;
    float coxa_length;
    float tibia_length;
    float femur_length; // 200

    float tibia_absolute_pos;
    float tibia_absolute_neg;

    float max_angle_coxa;
    float min_angle_coxa;
    float max_angle_coxa_w_margin;
    float min_angle_coxa_w_margin;
    float max_angle_tibia;
    float min_angle_tibia;
    float max_angle_femur;
    float min_angle_femur;
    float max_angle_femur_w_margin;
    float min_angle_femur_w_margin;
    float max_femur_to_gripper_dist;

    float positiv_saturated_femur[2];
    float negativ_saturated_femur[2];

    float min_femur_to_gripper_dist;
    float min_femur_to_gripper_dist_positive;

    float femur_overmargin_negative;
    float femur_overmargin_positive;

} LegDimensions;

typedef struct LegCompact {
  public:
    float body_angle;
    float body;
    float coxa_pitch;
    float max_angle_coxa;
    float min_angle_coxa;
    float coxa_length;
    float min_angle_femur;
    float tibia_absolute_pos;

    Circle inner;
    Circle outer;
    Circle fromabove_neg;
    Circle fromabove_pos;
    Circle winglet_neg;
    Circle winglet_pos;

    Intersect bot_inner_fromabove;
    Intersect bot_winglet_fromabove;
    Intersect top_inner_winglet;
    Intersect top_fromabove_winglet;
} LegCompact;
