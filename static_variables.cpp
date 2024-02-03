#include "static_variables.h"

LegDimensions get_SCARE_leg(float body_angle) {

    float coxa_margin = 0.0f;
    float femur_margin = 0.0f;
    float tibia_margin = 0.0f;
    float dist_margin = 0.0f;

    LegDimensions scare{};
    scare.body_angle = body_angle;
    scare.body = 185.0f;
    scare.coxa_angle_deg = 60.0f;
    scare.coxa_length = 165.0f;
    scare.femur_angle_deg = 90.0f; // 90
    scare.femur_length = 190.0f;
    scare.tibia_angle_deg = 120.0f; // 120
    scare.tibia_length = 200.0f;    // 200

    scare.max_angle_coxa = pI / 180.0f * scare.coxa_angle_deg;
    scare.min_angle_coxa = -pI / 180.0f * scare.coxa_angle_deg;
    scare.max_angle_coxa_w_margin =
        pI / 180.0f * (scare.coxa_angle_deg - coxa_margin);
    scare.min_angle_coxa_w_margin =
        -pI / 180.0f * (scare.coxa_angle_deg - coxa_margin);
    scare.max_angle_femur = pI / 180.0f * scare.femur_angle_deg;
    scare.min_angle_femur = -pI / 180.0f * scare.femur_angle_deg;
    scare.max_angle_femur_w_margin =
        pI / 180.0f * (scare.femur_angle_deg + femur_margin);
    scare.min_angle_femur_w_margin =
        -pI / 180.0f * (scare.femur_angle_deg + femur_margin);
    scare.max_angle_tibia = pI / 180.0f * scare.tibia_angle_deg;
    scare.min_angle_tibia = -pI / 180.0f * scare.tibia_angle_deg;

    scare.max_femur_to_gripper_dist = scare.tibia_length + scare.femur_length;

    scare.fem_tib_min[0] =
        scare.femur_length + scare.tibia_length * cos(scare.max_angle_tibia);
    scare.fem_tib_min[1] = scare.tibia_length * sin(scare.max_angle_tibia);

    scare.min_femur_to_gripper_dist =
        sqrt(scare.fem_tib_min[0] * scare.fem_tib_min[0] +
             scare.fem_tib_min[1] * scare.fem_tib_min[1]);
    scare.middle_TG =
        (scare.max_femur_to_gripper_dist + scare.min_femur_to_gripper_dist) /
        2.0f;
    scare.middle_TG_radius =
        (scare.max_femur_to_gripper_dist - scare.min_femur_to_gripper_dist) /
        2.0f;
    scare.middle_TG_radius_w_margin = scare.middle_TG_radius - dist_margin;

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
    /* scare.femur_overmargin = */
    /*     -scare.max_angle_femur + */
    /*     atan2f(scare.femur_length * sinf(scare.max_angle_femur) + */
    /*                (scare.tibia_length - dist_margin) * */
    /*                    sinf(scare.max_angle_femur + scare.max_angle_tibia), */
    /*            scare.femur_length * cosf(scare.max_angle_femur) + */
    /*                (scare.tibia_length - dist_margin) * */
    /*                    cosf(scare.max_angle_femur + scare.max_angle_tibia)); */
    scare.femur_overmargin =
        (scare.min_femur_to_gripper_dist * scare.min_femur_to_gripper_dist +
         scare.tibia_length * scare.tibia_length +
         scare.femur_length * scare.femur_length) /
        (2 * scare.min_femur_to_gripper_dist * scare.femur_length);

    return scare;
}
