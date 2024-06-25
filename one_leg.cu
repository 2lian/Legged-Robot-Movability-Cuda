#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "circles.cu.h"
#include "octree_util.cu.h"
#include "one_leg.cu.h"
#include "settings.h"

template <bool Reverse = false>
__host__ __device__ __forceinline__ void place_over_coxa(float3& coordinates,
                                                         const LegDimensions dim) {
    // Coxa as the frame of reference without rotation
    if constexpr (not Reverse)
        coordinates.x -= dim.body;
    float sin_memory;
    float cos_memory;
    if constexpr (Reverse)
        sincosf(dim.coxa_pitch, &sin_memory, &cos_memory);
    else
        sincosf(-dim.coxa_pitch, &sin_memory, &cos_memory);
    float buffer = coordinates.x * sin_memory;
    coordinates.x = coordinates.x * cos_memory - coordinates.z * sin_memory;
    coordinates.z = buffer + coordinates.z * cos_memory;
}

__host__ __device__ __forceinline__ float find_coxa_angle(const float3 coordinates) {
    // finding coxa angle
    return atan2f(coordinates.y, coordinates.x);
}

__host__ __device__ __forceinline__ void distance_to_circumf(const Circle circle, float x,
                                                             float y, float& distance,
                                                             bool& validity) {
    x -= circle.x;
    y -= circle.y;
    float magnitude = sqrtf(x * x + y * y);
    distance = circle.radius - magnitude; // if ouside the distance will be negative
    bool inside_the_circle = !signbit(distance); // will be true if distance is +
    validity =
        (inside_the_circle == circle.attractivity) or (fabs(distance) < CIRCLE_MARGIN);
}
__host__ __device__ __forceinline__ void force_clamp_on_circle(const Circle circle,
                                                               float& x, float& y,
                                                               float& distance,
                                                               bool& validity) {
    x -= circle.x;
    y -= circle.y;
    float magnitude = sqrtf(x * x + y * y);
    distance = circle.radius - magnitude;        // if inside the distance will be +
    bool inside_the_circle = !signbit(distance); // will be true if distance is +
    validity =
        (inside_the_circle == circle.attractivity) or fabs(distance) < CIRCLE_MARGIN;

    if (magnitude < CIRCLE_MARGIN) {
        x = 1;
        y = 0;
        magnitude = 1;
    }

    float radius_pre_normailed = circle.radius / magnitude;
    x = circle.x + x * radius_pre_normailed;
    y = circle.y + y * radius_pre_normailed;
}

template <bool OnlyCircles = false> // if the array has no points a chek is skipped
__host__ __device__ __forceinline__ bool
multi_circle_validate(float x, float y, Circle* circleArr, int number_of_circles) {
    for (int i = 0; i < number_of_circles; i++) {
        Circle& circle = circleArr[i];
        if constexpr (not OnlyCircles) {
            const bool is_point = abs(circle.radius) < CIRCLE_MARGIN;
            if (is_point) { // must skip points
                if constexpr (CIRCLE_ARR_ORDERED) {
                    return true;
                } else {
                    continue;
                }
            }
        }

        float distance;
        bool is_valid;
        distance_to_circumf(circle, x, y, distance, is_valid);
        if (not is_valid) {
            return false;
        }
    }
    return true; // return true is all valid
}

__host__ __device__ __forceinline__ bool multi_circle_clamp(float& x, float& y, Circle* circleArr,
                                                   const int number_of_circles) {
    // number_of_circles = 1;
    bool overall_validity = true; // switches to false if one result is false
    float potential_x = 0;
    float potential_y = 0;
    float previous_distance = 999999999999999.9;

    for (int i = 0; i < number_of_circles; i++) {
        Circle& circle = circleArr[i];
        float distance;
        bool validity;
        float x_onthe_circle = x;
        float y_onthe_circle = y;
        force_clamp_on_circle(circle, x_onthe_circle, y_onthe_circle, distance, validity);

        bool clamp_is_valid;
        bool is_point = abs(circle.radius) < CIRCLE_MARGIN;
        if (is_point) {
            if (overall_validity) { // if origin is valid, we do not clamp on ponts
                if constexpr (CIRCLE_ARR_ORDERED) {
                    break;
                } else {
                    continue;
                }
            }
            overall_validity = overall_validity;
            clamp_is_valid = true;
        } else {
            if constexpr (not MegaClamp) {
                if constexpr (CIRCLE_ARR_ORDERED) {
                    clamp_is_valid = multi_circle_validate<true>(
                        x_onthe_circle, y_onthe_circle, circleArr, MAX_CIRCLES);
                } else {
                    clamp_is_valid = multi_circle_validate(x_onthe_circle, y_onthe_circle,
                                                           circleArr, number_of_circles);
                }
            } else
                clamp_is_valid = true;
            overall_validity = overall_validity && validity;
            // clamp_is_valid = false;
        }
        bool closer_boundary = abs(previous_distance) > abs(distance);

        if (clamp_is_valid && closer_boundary) {
            // found a closer valid boundary
            previous_distance = distance;
            potential_x = x_onthe_circle;
            potential_y = y_onthe_circle;
        }
    }
    x -= potential_x;
    y -= potential_y;
    return overall_validity;
}
__host__ __device__ __forceinline__ void cancel_coxa_rotation(float3& coordinates,
                                                              const float coxa_angle,
                                                              float& cosine_coxa_memory,
                                                              float& sin_coxa_memory) {
    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    sincosf(-coxa_angle, &sin_coxa_memory, &cosine_coxa_memory);
    float buffer = coordinates.x * sin_coxa_memory;
    coordinates.x = coordinates.x * cosine_coxa_memory - coordinates.y * sin_coxa_memory;
    coordinates.y = buffer + coordinates.y * cosine_coxa_memory;
}

__host__ __device__ __forceinline__ void restore_coxa_rotation(float3& coordinates,
                                                               float cosine_coxa_memory,
                                                               float sin_coxa_memory) {

    float buffer = coordinates.y * sin_coxa_memory;
    coordinates.y = -coordinates.x * sin_coxa_memory + coordinates.y * cosine_coxa_memory;
    coordinates.x = coordinates.x * cosine_coxa_memory + buffer;
}

template <int UseCase = REACH_USECASE>
__host__ __device__ __forceinline__ bool eval_plane_circles(float& x, float& y,
                                                            const LegDimensions dim) {
    static_assert(UseCase == REACH_USECASE || UseCase == DIST_USECASE, "bad usecase");
    // Femur as the frame of reference witout rotation
    x -= dim.coxa_length;

    auto region = find_region(x, y, dim);
    constexpr uchar number_of_circles =
        (UseCase == REACH_USECASE) ? MAX_CIRCLES : MAX_CIRCLE_INTER;

    Circle circle_list[number_of_circles];
    auto* tail = circle_list;
    tail = insert_circles(dim, region, tail);

    bool is_valid;
    if constexpr (UseCase == REACH_USECASE) { // dircetly validates
        uchar stack_length = tail - circle_list;
        is_valid = multi_circle_validate<true>(x, y, circle_list, stack_length);
    } else if constexpr (UseCase == DIST_USECASE) {
        // __shared__ Circle sinter[MAX_CIRCLE_INTER];
        // __shared__ Circle* s_tail;
        // if (threadIdx.x == 0){
        //     s_tail = insert_intersecv2(dim, sinter);
        // }
        // __syncthreads();
        // auto tail_int = tail;
        // for (auto scp = sinter; scp < s_tail; scp++) {
        //     tail_int[0].x = scp[0].x;
        //     tail_int[0].y = scp[0].y;
        //     tail_int[0].radius = scp[0].radius;
        //     tail_int++;
        // }
        auto tail_int = insert_intersecv2(dim, tail); // adds intersection points

        uchar stack_length = tail_int - circle_list;
        is_valid = multi_circle_clamp(x, y, circle_list, stack_length);
        // validates and computes distance
    }

    return is_valid;
}

__host__ __device__ __forceinline__ bool place_on_vert_plane_circle(float& x, float& z,
                                                           const LegDimensions& dim) {
    return eval_plane_circles<DIST_USECASE>(x, z, dim);
}

template <typename Tout = void, // function vert plane is templated
          Tout (*vert_plane_func)(float&, float&,
                                  const LegDimensions&) = place_on_vert_plane_circle>
__host__ __device__ __forceinline__ Tout finish_finding_closest(float3& coordinates,
                                                                const LegDimensions& dim,
                                                                const float& coxa_angle) {
    // saturating coxa angle for dist
    bool coxaMegaSaturated = coxa_angle > (dim.max_angle_coxa + PI / 2) or
                             coxa_angle < (dim.min_angle_coxa - PI / 2);
    float saturated_coxa_angle;
    if (coxaMegaSaturated) {
        saturated_coxa_angle = (coxa_angle > 0) ? coxa_angle - PI : coxa_angle + PI;
    } else {
        saturated_coxa_angle =
            fmaxf(fminf(coxa_angle, dim.max_angle_coxa), dim.min_angle_coxa);
    }
    bool coxa_saturated = saturated_coxa_angle != coxa_angle;
    float coxa_limit = (coxa_angle > (dim.max_angle_coxa + dim.min_angle_coxa) / 2)
                           ? dim.max_angle_coxa
                           : dim.min_angle_coxa;

    float cosine_coxa_memory;
    float sin_coxa_memory;
    cancel_coxa_rotation(coordinates, saturated_coxa_angle, cosine_coxa_memory,
                         sin_coxa_memory);

    static_assert(std::is_same_v<Tout, bool> or std::is_same_v<Tout, void>,
                  "Type not supported");
    if constexpr (std::is_same_v<Tout, void>) {
        vert_plane_func(coordinates.x, coordinates.z, dim);
        restore_coxa_rotation(coordinates, cosine_coxa_memory, sin_coxa_memory);
        return;

    } else if (std::is_same_v<Tout, bool>) {
        Tout was_valid;
        float3 save = coordinates;
        // if (coxaMegaSaturated) {
        // coordinates.x = 0.01; // clamp on coxa axis
        // }
        was_valid = vert_plane_func(coordinates.x, coordinates.z, dim);
        // if (coxaMegaSaturated){
        // coordinates.x += save.x; // adds back coxa axis displacement
        // }
        if (was_valid and not coxaMegaSaturated) {

            float cos_coxa_memory2;
            float sin_coxa_memory2;
            cancel_coxa_rotation(save, coxa_limit - saturated_coxa_angle,
                                 cos_coxa_memory2, sin_coxa_memory2);
            save.x = 0;
            save.z = 0;
            const auto distance_clamped =
                linorm(coordinates);
            float distance_to_coxa_lim = linorm(save);
            const bool better_not_clamp = distance_clamped > distance_to_coxa_lim;
            if (better_not_clamp) {
                restore_coxa_rotation(save, cos_coxa_memory2, sin_coxa_memory2);
                coordinates = save;
            }
        }
        restore_coxa_rotation(coordinates, cosine_coxa_memory, sin_coxa_memory);
        return was_valid and not coxa_saturated;
    }
}

__host__ __device__ __inline__ bool reachability_circles(const float3& point,
                                                         const LegDimensions& dim)
// the tibia cannot exceed a specified cangle relative to the BODY
// this ensures that the tibia is for example always pointing down
{
    // Coxa as the frame of reference without rotation
    float3 result;
    result = point;
    place_over_coxa(result, dim);
    float required_angle_coxa;
    {
        const bool flip_flag = -signbit(result.x);
        if (flip_flag) {
            result.x *= -1;
            result.y *= -1;
        }

        // finding coxa angle
        required_angle_coxa = find_coxa_angle(result);
        if (flip_flag) {
            result.x *= -1;
            result.y *= -1;
        }
    }

    const bool coxa_is_valid = (required_angle_coxa > dim.max_angle_coxa) ||
                               (required_angle_coxa < dim.min_angle_coxa);
    if (coxa_is_valid) {
        return false;
    }

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox;
    float sin_angle_cox;
    cancel_coxa_rotation(result, required_angle_coxa, cos_angle_cox, sin_angle_cox);

    const bool reachability = eval_plane_circles<REACH_USECASE>(result.x, result.z, dim);
    return reachability;
}

__host__ __device__ __forceinline__ bool distance_circles(float3& result,
                                                          const LegDimensions& dim) {
    float3 closest = result;
    bool constexpr Reverse = true;
    place_over_coxa<not Reverse>(closest, dim);
    float3 closest_flip = closest;

    float coxangle = find_coxa_angle(closest);
    float coxangle_flip = (coxangle > 0) ? coxangle - pIgpu : coxangle + pIgpu;

    bool res = finish_finding_closest<bool>(closest, dim, coxangle);
    bool resflip = finish_finding_closest<bool>(closest_flip, dim, coxangle_flip);

    bool use_direct = (not(res ^ resflip)) ? linorm(closest) < linorm(closest_flip) : res;

    float3* result_to_use = (use_direct) ? &closest : &closest_flip;

    result = *result_to_use;
    place_over_coxa<Reverse>(result, dim);
    return res or resflip;
}

__global__ void reachability_circles_kernel(const Array<float3> input,
                                            const LegDimensions dim,
                                            Array<bool> const output) {
    // __shared__ LegDimensions chached_dim;
    // if (threadIdx.x == 0) {
    //     chached_dim = dim;
    // }
    // __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = reachability_circles(input.elements[i], dim);
    }
}

__global__ void distance_circles_kernel(const Array<float3> input,
                                        const LegDimensions dim,
                                        Array<float3> const output) {
    // __shared__ LegDimensions chached_dim;
    // if (threadIdx.x == 0) {
    //     chached_dim = dim;
    // }
    // __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        float3 result = input.elements[i];
        bool reachability = distance_circles(result, dim);
        output.elements[i] = result;
    }
}

__device__ inline float3 forward_kinematics(const float coxa, const float femur,
                                            const float tibia, const LegDimensions dim) {
    float3 result{0, 0, 0};
    result.x += dim.body;
    float cos_horiz, sin_horiz;
    sincosf(coxa, &sin_horiz, &cos_horiz);
    result.x += cos_horiz * dim.coxa_length;
    result.y += sin_horiz * dim.coxa_length;

    float cos, sin;
    sincosf(femur, &sin, &cos);
    float horiz_distance = cos * dim.femur_length;
    float vert_distance = sin * dim.femur_length;
    result.x += cos_horiz * horiz_distance;
    result.y += sin_horiz * horiz_distance;
    result.z += vert_distance;

    sincosf(tibia + femur, &sin, &cos);
    horiz_distance = cos * dim.tibia_length;
    vert_distance = sin * dim.tibia_length;
    result.x += cos_horiz * horiz_distance;
    result.y += sin_horiz * horiz_distance;
    result.z += vert_distance;

    return result;
};

__global__ void forward_kine_kernel(const Array<float3> angles_3_input,
                                    const LegDimensions dim,
                                    Array<float3> const output_xyz) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < angles_3_input.length; i += stride) {
        output_xyz.elements[i] =
            forward_kinematics(angles_3_input.elements[i].x, angles_3_input.elements[i].y,
                               angles_3_input.elements[i].z, dim);
    }
}
