#include "static_variables.h"
#include <cmath>
#include <iostream>
// #include <iostream>

LegDimensions leg_factory(float azimut, float body2coxa, float coxa_pitch_deg,
                          float coxa2tibia, float tibia2femur, float femur2tip,
                          float coxa_margin, float femur_margin, float tibia_margin,
                          float coxa_angle_deg, float femur_angle_deg,
                          float tibia_angle_deg, float dist_margin, float tib_abs_pos,
                          float tib_abs_neg) {

    LegDimensions leg{};
    leg.coxa_pitch = coxa_pitch_deg / 180.f * pI;
    leg.body = body2coxa;
    leg.coxa_length = coxa2tibia;
    leg.femur_length = tibia2femur;
    leg.tibia_length = femur2tip;
    leg.tibia_absolute_pos = tib_abs_pos / 180.0f * pI - leg.coxa_pitch;
    leg.tibia_absolute_neg = (-180.0f - tib_abs_neg) / 180.0f * pI - leg.coxa_pitch;
    // std::cout << leg.tibia_absolute_pos * 180 / pI << std::endl;
    // std::cout << leg.tibia_absolute_neg * 180 / pI << std::endl;

    leg.max_angle_coxa = pI / 180.0f * coxa_angle_deg;
    leg.min_angle_coxa = -pI / 180.0f * coxa_angle_deg;
    leg.max_angle_femur = pI / 180.0f * femur_angle_deg;
    leg.min_angle_femur = -pI / 180.0f * femur_angle_deg;
    leg.max_angle_tibia = pI / 180.0f * tibia_angle_deg;
    leg.min_angle_tibia = -pI / 180.0f * tibia_angle_deg;

    leg.body_angle = azimut;

    float fem_tib_min[2];
    float fem_tib_max[2];

    fem_tib_min[0] = leg.femur_length + leg.tibia_length * cos(leg.min_angle_tibia);
    fem_tib_min[1] = leg.tibia_length * sin(leg.min_angle_tibia);

    fem_tib_max[0] = leg.femur_length + leg.tibia_length * cos(leg.max_angle_tibia);
    fem_tib_max[1] = leg.tibia_length * sin(leg.max_angle_tibia);
    return leg;
}

LegDimensions get_moonbot_leg(float azimut) {
    float body2coxa = 181;
    float coxa_pitch_deg = 0;
    float coxa2tibia = 65.5;
    float tibia2femur = 129;
    float femur2tip = 160;

    float coxa_margin = 0.0f;
    float femur_margin = 0.0f;
    float tibia_margin = 0.0f;

    float coxa_angle_deg = 60.0f;
    float femur_angle_deg = 90.0f;  // 90
    float tibia_angle_deg = 120.0f; // 120

    float dist_margin = 0.0f;
    float tib_abs_pos = -5;
    float tib_abs_neg = -5;

    return leg_factory(azimut, body2coxa, coxa_pitch_deg, coxa2tibia, tibia2femur,
                       femur2tip, coxa_margin, femur_margin, tibia_margin, coxa_angle_deg,
                       femur_angle_deg, tibia_angle_deg, dist_margin, tib_abs_pos,
                       tib_abs_neg);
}

LegDimensions get_M2_leg(float azimut) {

    float body2coxa = 181;
    float coxa_pitch_deg = -45;
    float coxa2tibia = 65.5;
    float tibia2femur = 129;
    float femur2tip = 135;

    float coxa_margin = 0.0f;
    float femur_margin = 0.0f;
    float tibia_margin = 0.0f;

    float coxa_angle_deg = 60.0f;
    float femur_angle_deg = 90.0f;  // 90
    float tibia_angle_deg = 120.0f; // 120

    float dist_margin = 0.0f;
    float tib_abs_pos = -5;
    float tib_abs_neg = -5;

    return leg_factory(azimut, body2coxa, coxa_pitch_deg, coxa2tibia, tibia2femur,
                       femur2tip, coxa_margin, femur_margin, tibia_margin, coxa_angle_deg,
                       femur_angle_deg, tibia_angle_deg, dist_margin, tib_abs_pos,
                       tib_abs_neg);
}
