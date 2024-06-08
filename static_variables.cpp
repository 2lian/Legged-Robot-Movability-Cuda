#include "static_variables.h"
#include <cmath>
#include <iostream>

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

    leg.max_angle_coxa = pI / 180.0f * coxa_angle_deg;
    leg.min_angle_coxa = -pI / 180.0f * coxa_angle_deg;
    leg.max_angle_femur = pI / 180.0f * femur_angle_deg;
    leg.min_angle_femur = -pI / 180.0f * femur_angle_deg;
    leg.max_angle_tibia = pI / 180.0f * tibia_angle_deg;
    leg.min_angle_tibia = -pI / 180.0f * tibia_angle_deg;

    leg.body_angle = azimut;
    leg.max_angle_coxa_w_margin = pI / 180.0f * (coxa_angle_deg - coxa_margin);
    leg.min_angle_coxa_w_margin = -pI / 180.0f * (coxa_angle_deg - coxa_margin);
    leg.max_angle_femur_w_margin = pI / 180.0f * (femur_angle_deg + femur_margin);
    leg.min_angle_femur_w_margin = -pI / 180.0f * (femur_angle_deg + femur_margin);

    leg.max_femur_to_gripper_dist = leg.tibia_length + leg.femur_length;

    float fem_tib_min[2];
    float fem_tib_max[2];

    fem_tib_min[0] = leg.femur_length + leg.tibia_length * cos(leg.min_angle_tibia);
    fem_tib_min[1] = leg.tibia_length * sin(leg.min_angle_tibia);

    fem_tib_max[0] = leg.femur_length + leg.tibia_length * cos(leg.max_angle_tibia);
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

    leg.positiv_saturated_femur[0] = cos(leg.max_angle_femur) * leg.femur_length;
    leg.positiv_saturated_femur[1] = sin(leg.max_angle_femur) * leg.femur_length;

    leg.negativ_saturated_femur[0] = cos(leg.min_angle_femur) * leg.femur_length;
    leg.negativ_saturated_femur[1] = sin(leg.min_angle_femur) * leg.femur_length;

    // this is the maximum angle possiblefrom the femur to the target
    // after accounting for the tibia's angle saturation
    // So there's the femur angle, then this one added to it.
    leg.femur_overmargin_negative =
        acos((leg.min_femur_to_gripper_dist * leg.min_femur_to_gripper_dist +
              leg.femur_length * leg.femur_length - leg.tibia_length * leg.tibia_length) /
             (2 * leg.min_femur_to_gripper_dist * leg.femur_length));
    leg.femur_overmargin_positive =
        acos((leg.min_femur_to_gripper_dist * leg.min_femur_to_gripper_dist +
              leg.femur_length * leg.femur_length - leg.tibia_length * leg.tibia_length) /
             (2 * leg.min_femur_to_gripper_dist * leg.femur_length));
    Circle circle_list[10];
    // circles definitions
    Circle& too_close = circle_list[0];
    {
        Circle& the_circle = too_close;
        the_circle.x = 0;
        the_circle.y = 0;
        the_circle.radius = leg.min_femur_to_gripper_dist;
        the_circle.attractivity = false;
    }

    Circle& too_far = circle_list[1];
    {
        Circle& the_circle = too_far;
        the_circle.x = 0;
        the_circle.y = 0;
        the_circle.radius = leg.max_femur_to_gripper_dist;
        the_circle.attractivity = true;
    }

    Circle& negativ_winglet = circle_list[2];
    {
        Circle& the_circle = negativ_winglet;
        the_circle.x = leg.negativ_saturated_femur[0];
        the_circle.y = leg.negativ_saturated_femur[1];
        the_circle.radius = leg.tibia_length;
        the_circle.attractivity = true;
    }

    Circle& positiv_winglet = circle_list[3];
    {
        Circle& the_circle = positiv_winglet;
        the_circle.x = leg.positiv_saturated_femur[0];
        the_circle.y = leg.positiv_saturated_femur[1];
        the_circle.radius = leg.tibia_length;
        the_circle.attractivity = false;
    }

    Circle& from_above_negat = circle_list[4];
    {
        Circle& the_circle = from_above_negat;
        the_circle.x = leg.tibia_length * cos(leg.tibia_absolute_neg);
        the_circle.y = leg.tibia_length * sin(leg.tibia_absolute_neg);
        the_circle.radius = leg.femur_length;
        the_circle.attractivity = false;
    }

    Circle& from_above_pos = circle_list[5];
    {
        Circle& the_circle = from_above_pos;
        the_circle.x = leg.tibia_length * cos(leg.tibia_absolute_pos);
        the_circle.y = leg.tibia_length * sin(leg.tibia_absolute_pos);
        the_circle.radius = leg.femur_length;
        the_circle.attractivity = true;
    }

    Circle& intersect_04 = circle_list[6]; // inner and from above neg
    {
        Circle& the_circle = intersect_04;
     // tibia saturates and femur makes it so tibia reaches the from above limit
        float fem_angle = leg.tibia_absolute_neg - leg.min_angle_tibia;
        float x_fem = leg.femur_length * cos(fem_angle);
        float y_fem = leg.femur_length * sin(fem_angle);
        float tib_angle = leg.min_angle_tibia + fem_angle;
        float x_tib = leg.femur_length * cos(tib_angle);
        float y_tib = leg.femur_length * sin(tib_angle);
        the_circle.x = x_fem + x_tib;
        the_circle.y = y_fem + y_tib;
        the_circle.radius = 0;
        the_circle.attractivity = true;
    }

    Circle& intersect_24 = circle_list[7]; // winglet and from above neg
    {
        Circle& the_circle = intersect_24;
     // tibia at the from above limit and femur saturates
        float fem_angle = leg.min_angle_femur;
        float x_fem = leg.femur_length * cos(fem_angle);
        float y_fem = leg.femur_length * sin(fem_angle);
        float tib_angle = leg.tibia_absolute_neg;
        float x_tib = leg.femur_length * cos(tib_angle);
        float y_tib = leg.femur_length * sin(tib_angle);
        the_circle.x = x_fem + x_tib;
        the_circle.y = y_fem + y_tib;
        the_circle.radius = 0;
        the_circle.attractivity = true;
    }

    Circle& intersect_03 = circle_list[8]; // inner and winglet
    {
        Circle& the_circle = intersect_03;
                                              // tibia and femur saturates
        float fem_angle = leg.max_angle_femur;
        float x_fem = leg.femur_length * cos(fem_angle);
        float y_fem = leg.femur_length * sin(fem_angle);
        float tib_angle = leg.min_angle_tibia;
        float x_tib = leg.femur_length * cos(tib_angle);
        float y_tib = leg.femur_length * sin(tib_angle);
        the_circle.x = x_fem + x_tib;
        the_circle.y = y_fem + y_tib;
        the_circle.radius = 0;
        the_circle.attractivity = true;
    }

    Circle& intersect_35 = circle_list[9]; // winglet and  from above pos
    {
        Circle& the_circle = intersect_35;
    if (leg.max_angle_tibia + leg.min_angle_femur < leg.tibia_absolute_pos) {
        // saturated femur and limit from above tibia
        float fem_angle = leg.max_angle_femur;
        float x_fem = leg.femur_length * cos(fem_angle);
        float y_fem = leg.femur_length * sin(fem_angle);
        float tib_angle = leg.tibia_absolute_pos;
        float x_tib = leg.femur_length * cos(tib_angle);
        float y_tib = leg.femur_length * sin(tib_angle);
        the_circle.x = x_fem + x_tib;
        the_circle.y = y_fem + y_tib;
        the_circle.radius = 0;
        the_circle.attractivity = true;

    } else {
        // saturated tibia and femur making it reach the from above limit
        float fem_angle = leg.tibia_absolute_pos - leg.min_angle_tibia;
        float x_fem = leg.femur_length * cos(fem_angle);
        float y_fem = leg.femur_length * sin(fem_angle);
        float tib_angle = leg.min_angle_tibia + fem_angle;
        float x_tib = leg.femur_length * cos(tib_angle);
        float y_tib = leg.femur_length * sin(tib_angle);
        the_circle.x = x_fem + x_tib;
        the_circle.y = y_fem + y_tib;
        the_circle.radius = 0;
        the_circle.attractivity = true;
    }
    }

    std::cout << "circles:" << std::endl;
    for (int i = 0; i < 10; i += 1) {
        std::cout << circle_list[i].x << " , " << circle_list[i].y << std::endl;
    }
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

    return leg_factory(azimut, body2coxa, coxa_pitch_deg, coxa2tibia, tibia2femur,
                       femur2tip, coxa_margin, femur_margin, tibia_margin, coxa_angle_deg,
                       femur_angle_deg, tibia_angle_deg, dist_margin, tib_abs_pos,
                       tib_abs_neg);
}
