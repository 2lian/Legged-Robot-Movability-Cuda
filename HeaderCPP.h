#pragma once
#include <Eigen/Dense>

typedef struct RobotDimensions{
    public:
        float pI;
        float body;
        float coxa_angle_deg;
        float coxa_length;
        float tibia_angle_deg; //90
        float tibia_length;
        float tibia_length_squared;
        float femur_angle_deg; //120
        float femur_length; //200
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
        

        float fem_tib_min_host[2];

        float min_tibia_to_gripper_dist;
        float middle_TG;
        float middle_TG_radius;
        float middle_TG_radius_w_margin;

        float femur_overmargin;

} RobotDimensions;


int CalculateMedian(const Eigen::VectorXi& data);
float calculateMean(const float* arr, int size);
float calculateStdDev(const float* arr, int size, float mean);
