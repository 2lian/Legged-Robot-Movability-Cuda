#include "static_variables.h"
#include <cmath>

LegDimensions get_SCARE_leg(float body_angle) {

    float coxa_margin = 0.0f;
    float femur_margin = 0.0f;
    float tibia_margin = 0.0f;
    float dist_margin = 0.0f;

    LegDimensions scare{};
    scare.body = 181.0f;
    float coxa_angle_deg = 60.0f;
    scare.coxa_length = 65.5f;
    float femur_angle_deg = 95.0f; // 90
    scare.femur_length = 129.0f;
    float tibia_angle_deg = 110.0f; // 120
    scare.tibia_length = 160.0f;    // 200
    scare.tibia_absolute_pos = -0.0f / 180.0f * pI;
    scare.tibia_absolute_neg = -180.0f / 180.0f * pI;

    scare.max_angle_coxa = pI / 180.0f * coxa_angle_deg;
    scare.min_angle_coxa = -pI / 180.0f * coxa_angle_deg;
    scare.max_angle_femur = pI / 180.0f * femur_angle_deg;
    scare.min_angle_femur = -pI / 180.0f * femur_angle_deg;
    scare.max_angle_tibia = pI / 180.0f * tibia_angle_deg;
    scare.min_angle_tibia = -pI / 180.0f * tibia_angle_deg;

    scare.body_angle = body_angle;
    scare.max_angle_coxa_w_margin =
        pI / 180.0f * (coxa_angle_deg - coxa_margin);
    scare.min_angle_coxa_w_margin =
        -pI / 180.0f * (coxa_angle_deg - coxa_margin);
    scare.max_angle_femur_w_margin =
        pI / 180.0f * (femur_angle_deg + femur_margin);
    scare.min_angle_femur_w_margin =
        -pI / 180.0f * (femur_angle_deg + femur_margin);

    scare.max_femur_to_gripper_dist = scare.tibia_length + scare.femur_length;

    float fem_tib_min[2];
    float fem_tib_max[2];

    fem_tib_min[0] =
        scare.femur_length + scare.tibia_length * cos(scare.min_angle_tibia);
    fem_tib_min[1] = scare.tibia_length * sin(scare.min_angle_tibia);

    fem_tib_max[0] =
        scare.femur_length + scare.tibia_length * cos(scare.max_angle_tibia);
    fem_tib_max[1] = scare.tibia_length * sin(scare.max_angle_tibia);

    scare.min_femur_to_gripper_dist =
        sqrt(fem_tib_min[0] * fem_tib_min[0] + fem_tib_min[1] * fem_tib_min[1]);
    scare.min_femur_to_gripper_dist_positive =
        sqrt(fem_tib_max[0] * fem_tib_max[0] + fem_tib_max[1] * fem_tib_max[1]);
    // scare.middle_TG =
    //     (scare.max_femur_to_gripper_dist + scare.min_femur_to_gripper_dist) /
    //     2.0f;
    // scare.middle_TG_radius =
    //     (scare.max_femur_to_gripper_dist - scare.min_femur_to_gripper_dist) /
    //     2.0f;
    // scare.middle_TG_radius_w_margin = scare.middle_TG_radius - dist_margin;

    scare.positiv_saturated_femur[0] =
        cos(scare.max_angle_femur) * scare.femur_length;
    scare.positiv_saturated_femur[1] =
        sin(scare.max_angle_femur) * scare.femur_length;

    scare.negativ_saturated_femur[0] =
        cos(scare.min_angle_femur) * scare.femur_length;
    scare.negativ_saturated_femur[1] =
        sin(scare.min_angle_femur) * scare.femur_length;

    // this is the maximum angle possiblefrom the femur to the target
    // after accounting for the tibia's angle saturation
    // So there's the femur angle, then this one added to it.
    scare.femur_overmargin_negative = acos(
        (scare.min_femur_to_gripper_dist * scare.min_femur_to_gripper_dist +
         scare.femur_length * scare.femur_length -
         scare.tibia_length * scare.tibia_length) /
        (2 * scare.min_femur_to_gripper_dist * scare.femur_length));
    scare.femur_overmargin_positive = acos(
        (scare.min_femur_to_gripper_dist * scare.min_femur_to_gripper_dist +
         scare.femur_length * scare.femur_length -
         scare.tibia_length * scare.tibia_length) /
        (2 * scare.min_femur_to_gripper_dist * scare.femur_length));
    /* std::cout << scare.femur_overmargin << std::endl; */

    return scare;
}
