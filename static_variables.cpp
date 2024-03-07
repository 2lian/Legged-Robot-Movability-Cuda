#include "static_variables.h"
#include <cmath>

LegDimensions
leg_factory(float azimut, float body2coxa, float coxa_pitch_deg,
            float coxa2tibia, float tibia2femur, float femur2tip,
            float coxa_margin, float femur_margin, float tibia_margin,
            float coxa_angle_deg, float femur_angle_deg, float tibia_angle_deg,
            float dist_margin, float tib_abs_pos, float tib_abs_neg) {

    LegDimensions leg{};
    leg.coxa_pitch = coxa_pitch_deg / 180.f * pI;
    leg.body = body2coxa;
    leg.coxa_length = coxa2tibia;
    leg.femur_length = tibia2femur;
    leg.tibia_length = femur2tip;
    leg.tibia_absolute_pos = tib_abs_pos / 180.0f * pI - leg.coxa_pitch;
    leg.tibia_absolute_neg =
        (-180.0f - tib_abs_neg) / 180.0f * pI - leg.coxa_pitch;

    leg.max_angle_coxa = pI / 180.0f * coxa_angle_deg;
    leg.min_angle_coxa = -pI / 180.0f * coxa_angle_deg;
    leg.max_angle_femur = pI / 180.0f * femur_angle_deg;
    leg.min_angle_femur = -pI / 180.0f * femur_angle_deg;
    leg.max_angle_tibia = pI / 180.0f * tibia_angle_deg;
    leg.min_angle_tibia = -pI / 180.0f * tibia_angle_deg;

    leg.body_angle = azimut;
    leg.max_angle_coxa_w_margin = pI / 180.0f * (coxa_angle_deg - coxa_margin);
    leg.min_angle_coxa_w_margin = -pI / 180.0f * (coxa_angle_deg - coxa_margin);
    leg.max_angle_femur_w_margin =
        pI / 180.0f * (femur_angle_deg + femur_margin);
    leg.min_angle_femur_w_margin =
        -pI / 180.0f * (femur_angle_deg + femur_margin);

    leg.max_femur_to_gripper_dist = leg.tibia_length + leg.femur_length;

    float fem_tib_min[2];
    float fem_tib_max[2];

    fem_tib_min[0] =
        leg.femur_length + leg.tibia_length * cos(leg.min_angle_tibia);
    fem_tib_min[1] = leg.tibia_length * sin(leg.min_angle_tibia);

    fem_tib_max[0] =
        leg.femur_length + leg.tibia_length * cos(leg.max_angle_tibia);
    fem_tib_max[1] = leg.tibia_length * sin(leg.max_angle_tibia);

    leg.min_femur_to_gripper_dist =
        sqrt(fem_tib_min[0] * fem_tib_min[0] + fem_tib_min[1] * fem_tib_min[1]);
    leg.min_femur_to_gripper_dist_positive =
        sqrt(fem_tib_max[0] * fem_tib_max[0] + fem_tib_max[1] * fem_tib_max[1]);
    // scare.middle_TG =
    //     (scare.max_femur_to_gripper_dist + scare.min_femur_to_gripper_dist) /
    //     2.0f;
    // scare.middle_TG_radius =
    //     (scare.max_femur_to_gripper_dist - scare.min_femur_to_gripper_dist) /
    //     2.0f;
    // scare.middle_TG_radius_w_margin = scare.middle_TG_radius - dist_margin;

    leg.positiv_saturated_femur[0] =
        cos(leg.max_angle_femur) * leg.femur_length;
    leg.positiv_saturated_femur[1] =
        sin(leg.max_angle_femur) * leg.femur_length;

    leg.negativ_saturated_femur[0] =
        cos(leg.min_angle_femur) * leg.femur_length;
    leg.negativ_saturated_femur[1] =
        sin(leg.min_angle_femur) * leg.femur_length;

    // this is the maximum angle possiblefrom the femur to the target
    // after accounting for the tibia's angle saturation
    // So there's the femur angle, then this one added to it.
    leg.femur_overmargin_negative =
        acos((leg.min_femur_to_gripper_dist * leg.min_femur_to_gripper_dist +
              leg.femur_length * leg.femur_length -
              leg.tibia_length * leg.tibia_length) /
             (2 * leg.min_femur_to_gripper_dist * leg.femur_length));
    leg.femur_overmargin_positive =
        acos((leg.min_femur_to_gripper_dist * leg.min_femur_to_gripper_dist +
              leg.femur_length * leg.femur_length -
              leg.tibia_length * leg.tibia_length) /
             (2 * leg.min_femur_to_gripper_dist * leg.femur_length));

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
    float femur_angle_deg = 95.0f;  // 90
    float tibia_angle_deg = 110.0f; // 120

    float dist_margin = 0.0f;
    float tib_abs_pos = -5;
    float tib_abs_neg = -5;

    return leg_factory(azimut, body2coxa, coxa_pitch_deg, coxa2tibia,
                       tibia2femur, femur2tip, coxa_margin, femur_margin,
                       tibia_margin, coxa_angle_deg, femur_angle_deg,
                       tibia_angle_deg, dist_margin, tib_abs_pos, tib_abs_neg);
}

LegDimensions get_M2_leg(float azimut) {

    float body2coxa = 181;
    float coxa_pitch_deg = -45;
    float coxa2tibia = 65.5;
    float tibia2femur = 129;
    float femur2tip = 160;

    float coxa_margin = 0.0f;
    float femur_margin = 0.0f;
    float tibia_margin = 0.0f;

    float coxa_angle_deg = 60.0f;
    float femur_angle_deg = 95.0f;  // 90
    float tibia_angle_deg = 110.0f; // 120

    float dist_margin = 0.0f;
    float tib_abs_pos = -5;
    float tib_abs_neg = -5;

    return leg_factory(azimut, body2coxa, coxa_pitch_deg, coxa2tibia,
                       tibia2femur, femur2tip, coxa_margin, femur_margin,
                       tibia_margin, coxa_angle_deg, femur_angle_deg,
                       tibia_angle_deg, dist_margin, tib_abs_pos, tib_abs_neg);
}
