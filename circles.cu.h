#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "leg_geometry.cu.h"
#include "unified_math_cuda.cu.h"

#define NUMBER_OF_UPPER_CIRCLES 4
#define NUMBER_OF_MIDDLE_CIRCLES 4
#define NUMBER_OF_LOWER_CIRCLES 4
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_CIRCLES                                                                      \
    MAX(MAX(NUMBER_OF_UPPER_CIRCLES, NUMBER_OF_MIDDLE_CIRCLES), NUMBER_OF_LOWER_CIRCLES)
#define MAX_INTERSECT 4
#define MAX_CIRCLE_INTER (MAX_INTERSECT + MAX_CIRCLES)

#define LOWER_WINGLET_REGION 0
#define LOWER_FROMABOVE_REGION 1
#define MIDDLE_REGION 2
#define UPPER_REGION 3

// extern float sin(float x); // shuts up clangd
// extern float cos(float x);
// extern float atan2f(float x, float y);

__device__ __forceinline__ uchar find_region(float x, float z, const LegDimensions dim) {
    float femur_angle_raw = atan2f(z, x);
    bool lower_fromabove_region = (femur_angle_raw <= dim.tibia_absolute_neg);
    bool lower_winglet_region = (femur_angle_raw <= dim.min_angle_femur);
    // bool lower_region = false;
    bool upper_region = (femur_angle_raw >= dim.tibia_absolute_pos);
    uchar region;
    if (upper_region) {
        region = UPPER_REGION;
    } else if (lower_fromabove_region) {
        region = LOWER_FROMABOVE_REGION;
    } else if (lower_winglet_region) {
        region = LOWER_WINGLET_REGION;
    } else {
        region = MIDDLE_REGION;
    }
    return region;
}

__forceinline__ __host__ __device__ Circle inner_circle(const LegDimensions leg) {
    Circle too_close;
    too_close.x = 0;
    too_close.y = 0;
    too_close.radius = min_femur_to_gripper_dist(leg);
    too_close.attractivity = false;
    return too_close;
}

__forceinline__ __host__ __device__ Circle outer_circle(const LegDimensions leg) {
    Circle circle;
    circle.x = 0;
    circle.y = 0;
    circle.radius = max_femur_to_gripper_dist(leg);
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Circle fromabove_pos_circle(const LegDimensions leg) {
    Circle circle;
    circle.x = leg.tibia_length * cos(leg.tibia_absolute_pos);
    circle.y = leg.tibia_length * sin(leg.tibia_absolute_pos);
    circle.radius = leg.femur_length;
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Circle fromabove_neg_circle(const LegDimensions leg) {
    Circle circle;
    circle.x = leg.tibia_length * cos(leg.tibia_absolute_neg);
    circle.y = leg.tibia_length * sin(leg.tibia_absolute_neg);
    circle.radius = leg.femur_length;
    circle.attractivity = false;
    return circle;
}

__forceinline__ __host__ __device__ Circle winglet_pos_circle(const LegDimensions leg) {
    Circle circle;
    // circle.x = leg.positiv_saturated_femur[0];
    // circle.y = leg.positiv_saturated_femur[1];
    saturated_femur<UPPER_SIDE>(leg, circle.x, circle.y);
    circle.radius = leg.tibia_length;
    circle.attractivity = false;
    return circle;
}

__forceinline__ __host__ __device__ Circle winglet_neg_circle(const LegDimensions leg) {
    Circle circle;
    // circle.x = leg.negativ_saturated_femur[0];
    // circle.y = leg.negativ_saturated_femur[1];
    saturated_femur<LOWER_SIDE>(leg, circle.x, circle.y);
    circle.radius = leg.tibia_length;
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Circle
top_inner_winglet_intersect(const LegDimensions leg) {
    Circle circle;
    float fem_angle = leg.max_angle_femur;
    float x_fem = leg.femur_length * cos(fem_angle);
    float y_fem = leg.femur_length * sin(fem_angle);
    float tib_angle = leg.min_angle_tibia + fem_angle;
    float x_tib = leg.tibia_length * cos(tib_angle);
    float y_tib = leg.tibia_length * sin(tib_angle);
    circle.x = x_fem + x_tib;
    circle.y = y_fem + y_tib;
    circle.radius = 0;
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Circle
bot_inner_fromabove_intersect(const LegDimensions leg) {
    Circle circle;
    float fem_angle = leg.tibia_absolute_neg - leg.min_angle_tibia;
    float x_fem = leg.femur_length * cos(fem_angle);
    float y_fem = leg.femur_length * sin(fem_angle);
    float tib_angle = leg.min_angle_tibia + fem_angle;
    float x_tib = leg.tibia_length * cos(tib_angle);
    float y_tib = leg.tibia_length * sin(tib_angle);
    circle.x = x_fem + x_tib;
    circle.y = y_fem + y_tib;
    circle.radius = 0;
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Circle
bot_winglet_fromabove_intersect(const LegDimensions leg) {
    Circle circle;
    float fem_angle = leg.min_angle_femur;
    float x_fem = leg.femur_length * cos(fem_angle);
    float y_fem = leg.femur_length * sin(fem_angle);
    float tib_angle = leg.tibia_absolute_neg;
    float x_tib = leg.tibia_length * cos(tib_angle);
    float y_tib = leg.tibia_length * sin(tib_angle);
    circle.x = x_fem + x_tib;
    circle.y = y_fem + y_tib;
    circle.radius = 0;
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Circle
top_fromabove_winglet_intersect(const LegDimensions leg) {
    Circle circle;
    float fem_angle;
    float tib_angle;
    // if (leg.max_angle_tibia + leg.min_angle_femur < leg.tibia_absolute_pos) {
    // saturated femur and limit from above tibia
    fem_angle = leg.max_angle_femur;
    tib_angle = leg.tibia_absolute_pos;
    // } else {
    // saturated tibia and femur making it reach the from above limit
    // fem_angle = leg.tibia_absolute_pos - leg.min_angle_tibia;
    // tib_angle = leg.min_angle_tibia + fem_angle;
    // }
    float x_fem = leg.femur_length * cos(fem_angle);
    float y_fem = leg.femur_length * sin(fem_angle);
    float x_tib = leg.tibia_length * cos(tib_angle);
    float y_tib = leg.tibia_length * sin(tib_angle);
    circle.x = x_fem + x_tib;
    circle.y = y_fem + y_tib;
    circle.radius = 0;
    circle.attractivity = true;
    return circle;
}

__forceinline__ __host__ __device__ Cricle* insert_always_circle(const LegDimensions leg,
                                                                 Circle* head) {
    head[0] = inner_circle(leg);
    head[1] = fromabove_neg_circle(leg);
    head[2] = winglet_pos_circle(leg);
    auto tail = head + 1;
    return tail;
}

__forceinline__ __host__ __device__ Cricle* insert_lower_circle(const LegDimensions leg,
                                                                Circle* head) {
    head[0] = winglet_neg_circle(leg);
    head[1] = fromabove_neg_circle(leg);
    auto tail = head + 2;
    return tail;
}

__forceinline__ __host__ __device__ Cricle*
insert_lowerFromAbove_circle(const LegDimensions leg, Circle* head) {
    head[0] = winglet_neg_circle(leg);
    head[0].attractivity = false;
    head[1] = fromabove_neg_circle(leg);
    head[1].attractivity = true;
    auto tail = head + 2;
    return tail;
}

__forceinline__ __host__ __device__ Cricle* insert_middle_circle(const LegDimensions leg,
                                                                 Circle* head) {
    head[0] = outer_circle(leg);
    auto tail = head + 1;
    return tail;
}

__forceinline__ __host__ __device__ Cricle* insert_upper_circle(const LegDimensions leg,
                                                                Circle* head) {
    head[0] = fromabove_pos_circle(leg);
    head[1] = winglet_pos_circle(leg);
    auto tail = head + 2;
    return tail;
}

__host__ __device__ Cricle* insert_circles(const LegDimensions leg, uchar region,
                                           Circle* head) {
    // Don't forget to change MAX_CIRCLES if you touch this
    auto tail = insert_always_circle(leg, head);
    if (region == LOWER_WINGLET_REGION) {
        tail = insert_lower_circle(leg, tail);
    } else if (region == LOWER_FROMABOVE_REGION) {
        tail = insert_lowerFromAbove_circle(leg, tail);
    } else if (region == MIDDLE_REGION) {
        tail = insert_middle_circle(leg, tail);
    } else {
        tail = insert_upper_circle(leg, tail);
    }
    return tail;
}

__forceinline__ __host__ __device__ Cricle*
insert_lower_intersect(const LegDimensions leg, Circle* head) {
    head[0] = bot_inner_fromabove_intersect(leg);
    head[1] = bot_winglet_fromabove_intersect(leg);
    auto tail = head + 2;
    return tail;
}

__forceinline__ __host__ __device__ Cricle*
insert_middle_intersect(const LegDimensions leg, Circle* head) {
    head[0] = bot_inner_fromabove_intersect(leg);
    head[1] = top_inner_winglet_intersect(leg);
    auto tail = head + 2;
    return tail;
}

__forceinline__ __host__ __device__ Cricle*
insert_upper_intersect(const LegDimensions leg, Circle* head) {
    head[0] = top_inner_winglet_intersect(leg);
    head[1] = top_fromabove_winglet_intersect(leg);
    auto tail = head + 2;
    return tail;
}

__host__ __device__ Cricle* insert_intersec(const LegDimensions leg, uchar region,
                                            Circle* head) {
    // Don't forget to change MAX_INTERSECT if you touch this
    auto& tail = head;
    tail = insert_lower_intersect(leg, tail);
    tail = insert_upper_intersect(leg, tail);
    return tail;
}

__host__ LegCompact LegDim2LegComp(LegDimensions leg) {
    LegCompact c;
    c.body_angle = leg.body_angle;
    c.body = leg.body;
    c.coxa_pitch = leg.coxa_pitch;
    c.coxa_length = leg.coxa_length;
    c.min_angle_femur = leg.min_angle_femur;
    c.tibia_absolute_pos = leg.tibia_absolute_pos;
    c.max_angle_coxa = leg.max_angle_coxa;
    c.min_angle_coxa = leg.min_angle_coxa;

    c.inner = inner_circle(leg);
    c.outer = outer_circle(leg);
    c.fromabove_neg = fromabove_neg_circle(leg);
    c.fromabove_pos = fromabove_pos_circle(leg);
    c.winglet_neg = winglet_neg_circle(leg);
    c.winglet_pos = winglet_pos_circle(leg);

    c.bot_inner_fromabove = bot_inner_fromabove_intersect(leg);
    c.bot_winglet_fromabove = bot_winglet_fromabove_intersect(leg);
    c.top_inner_winglet = top_inner_winglet_intersect(leg);
    c.top_fromabove_winglet = top_fromabove_winglet_intersect(leg);
    return c;
}
