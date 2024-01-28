#pragma once

constexpr float pI =
    3.14159265358979323846264338327950288419716939937510582097f;

typedef struct LegDimensions {
  public:
    float body_angle;
    float body;
    float coxa_angle_deg;
    float coxa_length;
    float tibia_angle_deg; // 90
    float tibia_length;
    float tibia_length_squared;
    float femur_angle_deg; // 120
    float femur_length;    // 200
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
    float max_tibia_to_gripper_dist;

    float positiv_saturated_femur[2];

    float negativ_saturated_femur[2];

    float fem_tib_min[2];

    float min_tibia_to_gripper_dist;
    float middle_TG;
    float middle_TG_radius;
    float middle_TG_radius_w_margin;

    float femur_overmargin;

} LegDimensions;

LegDimensions get_SCARE_leg(float body_angle);
