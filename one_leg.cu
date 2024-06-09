#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "circles.cu.h"
#include "one_leg.cu.h"

#define CIRCLE_MARGIN 0.1
#define REACH_USECASE 0
#define DIST_USECASE 1

__device__ __forceinline__ void place_over_coxa(float3& coordinates,
                                                const LegDimensions& dim) {
    // Coxa as the frame of reference without rotation
    coordinates.x -= dim.body;
    float sin_coxa_memory;
    float cosine_coxa_memory;
    sincosf(-dim.coxa_pitch, &sin_coxa_memory, &cosine_coxa_memory);
    float buffer = coordinates.x * sin_coxa_memory;
    coordinates.x = coordinates.x * cosine_coxa_memory - coordinates.z * sin_coxa_memory;
    coordinates.z = buffer + coordinates.z * cosine_coxa_memory;
}

__device__ __forceinline__ float find_coxa_angle(const float3& coordinates) {
    // finding coxa angle
    return atan2f(coordinates.y, coordinates.x);
}

__device__ inline bool is_in_circle(const float* center, const float radius, float x,
                                    float y) {
    x -= center[0];
    y -= center[1];
    return sqrtf(x * x + y * y) < radius;
}

__device__ __forceinline__ void distance_to_circumf(Circle circle, float x, float y,
                                                    float& distance, bool& validity) {
    x -= circle.x;
    y -= circle.y;
    float magnitude = sqrtf(x * x + y * y);
    distance = circle.radius - magnitude; // if ouside the distance will be negative
    bool inside_the_circle = !signbit(distance); // will be true if distance is +
    validity =
        (inside_the_circle == circle.attractivity) or (fabs(distance) < CIRCLE_MARGIN);
}
__device__ __forceinline__ void force_clamp_on_circle(Circle circle, float& x, float& y,
                                                      float& distance, bool& validity) {
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

    x = circle.x + circle.radius * x / magnitude;
    y = circle.y + circle.radius * y / magnitude;
}
__device__ __forceinline__ void clamp_on_circle(const float* center, const float& radius,
                                                const bool& clamp_direction, float& x,
                                                float& y) {
    x -= center[0];
    y -= center[1];
    float magnitude = sqrtf(x * x + y * y);
    bool radius_direction = !signbit(radius - magnitude);

    if (clamp_direction != radius_direction or magnitude < 0.001f) {
        x = 0;
        y = 0;
        return;
    }

    x -= radius * x / magnitude;
    y -= radius * y / magnitude;
}

__device__ __forceinline__ void place_on_vert_plane(float& x, float& z,
                                                    const LegDimensions& dim) {
    // Femur as the frame of reference witout rotation
    x -= dim.coxa_length;

    float femur_angle_raw = atan2f(z, x);
    float femur_distance = norm3df(z, x, 0);

    bool positiv_side = !signbit(z);
    const float* saturated_femur_of_interest =
        (positiv_side) ? dim.positiv_saturated_femur : dim.negativ_saturated_femur;
    const float& tibia_saturation =
        (positiv_side) ? dim.max_angle_tibia : -dim.min_angle_tibia;
    const float& femur_saturation =
        (positiv_side) ? dim.max_angle_femur : -dim.min_angle_femur;

    float tibia_angle_raw =
        atan2f(z - saturated_femur_of_interest[1], x - saturated_femur_of_interest[0]);
    if (!positiv_side) {
        tibia_angle_raw *= -1;
        femur_angle_raw *= -1;
    }

    bool too_close = femur_distance < dim.min_femur_to_gripper_dist;

    bool femur_condition = (femur_angle_raw <= femur_saturation);
    bool femur_cond_wider =
        femur_angle_raw <= (femur_saturation + dim.femur_overmargin_negative);
    bool tibia_condition =
        (((tibia_angle_raw <= (tibia_saturation + femur_saturation)) and
          !signbit(tibia_angle_raw)) or
         ((tibia_angle_raw <= (tibia_saturation + femur_saturation - 2 * pIgpu))));

    float center[2] = {0, 0};
    float radius = dim.min_femur_to_gripper_dist;
    bool& clamp_direction = too_close;

    if (femur_condition and not too_close) { // implies femur is not saturated,
        // and point is not in the innermost circle, so we clamp onto the outermost circle
        radius = dim.max_femur_to_gripper_dist;
    } else if (tibia_condition and not too_close) { // implies tibia is not saturated
        // and point is not  is the innercircle
        // so we clamp on the  outer winglet
        center[0] = saturated_femur_of_interest[0];
        center[1] = saturated_femur_of_interest[1];
        radius = dim.tibia_length;
    } else if (not femur_cond_wider and not tibia_condition) {
        // implies we are not between the two winglets and tibia is saturated
        center[0] = cosf(dim.femur_overmargin_negative + femur_saturation) *
                    dim.min_femur_to_gripper_dist;
        center[1] = sinf(dim.femur_overmargin_negative + femur_saturation) *
                    dim.min_femur_to_gripper_dist;
        center[1] = copysignf(center[1], z);
        radius = 0;
        clamp_direction = false;
    };

    clamp_on_circle(center, radius, clamp_direction, x, z);
}

__device__ __forceinline__ void cancel_coxa_rotation(float3& coordinates,
                                                     const float& coxa_angle,
                                                     float& cosine_coxa_memory,
                                                     float& sin_coxa_memory) {
    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    sincosf(-coxa_angle, &sin_coxa_memory, &cosine_coxa_memory);
    float buffer = coordinates.x * sin_coxa_memory;
    coordinates.x = coordinates.x * cosine_coxa_memory - coordinates.y * sin_coxa_memory;
    coordinates.y = buffer + coordinates.y * cosine_coxa_memory;
}

__device__ __forceinline__ void restore_coxa_rotation(float3& coordinates,
                                                      float& cosine_coxa_memory,
                                                      float& sin_coxa_memory) {

    float buffer = coordinates.y * sin_coxa_memory;
    coordinates.y = -coordinates.x * sin_coxa_memory + coordinates.y * cosine_coxa_memory;
    coordinates.x = coordinates.x * cosine_coxa_memory + buffer;
}

template <typename Tout = void, // function vert plane is templated
          Tout (*vert_plane_func)(float&, float&,
                                  const LegDimensions&) = place_on_vert_plane>
__device__ __forceinline__ Tout finish_finding_closest(float3& coordinates,
                                                       const LegDimensions& dim,
                                                       const float& coxa_angle) {
    // saturating coxa angle for dist
    float saturated_coxa_angle = fmaxf(fminf(coxa_angle, dim.max_angle_coxa_w_margin),
                                       dim.min_angle_coxa_w_margin);
    bool coxa_saturated = saturated_coxa_angle != coxa_angle;
    float coxa_limit =
        (coxa_angle > (dim.max_angle_coxa_w_margin + dim.min_angle_coxa_w_margin) / 2)
            ? dim.max_angle_coxa_w_margin
            : dim.min_angle_coxa_w_margin;

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
        was_valid = vert_plane_func(coordinates.x, coordinates.z, dim);
        if (was_valid) {

            float cosine_coxa_memory2;
            float sin_coxa_memory2;
            cancel_coxa_rotation(save, coxa_limit - saturated_coxa_angle,
                                 cosine_coxa_memory2, sin_coxa_memory2);
            save.x = 0;
            save.z = 0;
            const auto distance_clamped =
                norm3df(coordinates.x, coordinates.y, coordinates.z);
            float distance_to_coxa_lim = norm3df(save.x, save.y, save.z);
            const bool better_not_clamp = distance_clamped > distance_to_coxa_lim;
            if (better_not_clamp) {
                restore_coxa_rotation(save, cosine_coxa_memory2, sin_coxa_memory2);
                coordinates = save;
            }
        }
        restore_coxa_rotation(coordinates, cosine_coxa_memory, sin_coxa_memory);
        return was_valid;
    }
}

__device__ __forceinline__ float3 dist_double_solf3(const float3& point,
                                                    const LegDimensions& dim) {
    float3 closest = point;
    place_over_coxa(closest, dim);
    float3 closest_flip = closest;

    float coxangle = find_coxa_angle(closest);
    float coxangle_flip = (coxangle > 0) ? coxangle - pIgpu : coxangle + pIgpu;

    finish_finding_closest(closest, dim, coxangle);
    finish_finding_closest(closest_flip, dim, coxangle_flip);

    float3* result_to_use = (norm3df(closest.x, closest.y, closest.z) <
                             norm3df(closest_flip.x, closest_flip.y, closest_flip.z))
                                ? &closest
                                : &closest_flip;

    return *result_to_use;
}

__device__ bool reachability_vect(const float3& point, const LegDimensions& dim)
// angle flipping on negative x values
{
    // Coxa as the frame of reference without rotation
    float3 result;
    result = point;
    place_over_coxa(result, dim);
    bool flip_flag = -signbit(result.x);

    if (flip_flag) {
        result.x *= -1;
        result.y *= -1;
    }

    // finding coxa angle
    float required_angle_coxa = find_coxa_angle(result);

    if (flip_flag) {
        result.x *= -1;
        result.y *= -1;
    }

    // flipping angle if above +-90deg

    if ((required_angle_coxa > dim.max_angle_coxa) ||
        (required_angle_coxa < dim.min_angle_coxa)) {
        return false;
    }

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox;
    float sin_angle_cox;
    cancel_coxa_rotation(result, required_angle_coxa, cos_angle_cox, sin_angle_cox);

    // Femur as the frame of reference witout rotation
    result.x -= dim.coxa_length;

    float linnorm = norm3df(result.x, result.y, result.z);

    if ((linnorm < dim.min_femur_to_gripper_dist) ||
        (linnorm > dim.max_femur_to_gripper_dist)) {
        return false;
    }

    // finding femur angle
    float required_angle_femur = atan2f(result.z, result.x);

    if ((required_angle_femur >= dim.min_angle_femur) &&
        (required_angle_femur <= dim.max_angle_femur)) {
        return true;
    }

    // distance to femur at the most extrem position, this value is pre_computed
    linnorm = fminf(norm3df(result.x - dim.positiv_saturated_femur[0], 0,
                            result.z - dim.positiv_saturated_femur[1]),
                    norm3df(result.x - dim.negativ_saturated_femur[0], 0,
                            result.z - dim.negativ_saturated_femur[1]));

    return linnorm <= dim.tibia_length;
}

__device__ bool reachability_absolute_tibia_limit(const float3& point,
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
        bool flip_flag = -signbit(result.x);

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

    if ((required_angle_coxa > dim.max_angle_coxa) ||
        (required_angle_coxa < dim.min_angle_coxa)) {
        return false;
    }

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox;
    float sin_angle_cox;
    cancel_coxa_rotation(result, required_angle_coxa, cos_angle_cox, sin_angle_cox);

    // Femur as the frame of reference witout rotation
    result.x -= dim.coxa_length;

    {
        float linnorm = norm3df(result.x, result.y, result.z);

        if ((linnorm < dim.min_femur_to_gripper_dist) ||
            (linnorm > dim.max_femur_to_gripper_dist)) {
            return false;
        }
    }

    bool in_negativ_tib_circle;
    bool in_positive_sat_circle;
    bool in_negative_sat_circle;
    bool inside_femur;
    bool in_positiv_tib_circle;

    {
        float deported_center_negativ[2];
        sincosf(dim.tibia_absolute_neg, &sin_angle_cox, &cos_angle_cox);
        deported_center_negativ[0] = dim.tibia_length * cos_angle_cox;
        deported_center_negativ[1] = dim.tibia_length * sin_angle_cox;
        // should not be inside
        in_negativ_tib_circle =
            norm3df(result.x - deported_center_negativ[0], 0,
                    result.z - deported_center_negativ[1]) <= dim.femur_length;
    }

    // finding femur angle
    {
        float required_angle_femur = atan2f(result.z, result.x);
        inside_femur = (required_angle_femur >= dim.min_angle_femur) &&
                       (required_angle_femur <= dim.tibia_absolute_pos);
    }
    {
        // - saturation is alowed
        in_negative_sat_circle =
            norm3df(result.x - dim.negativ_saturated_femur[0], 0,
                    result.z - dim.negativ_saturated_femur[1]) <= dim.tibia_length;
    }

    // distance to femur at the most extrem  femur position should be less than
    // tibia length to be in the saturation circle, circle center is
    // pre_computed

    {
        // We don´t want + saturation
        in_positive_sat_circle =
            norm3df(result.x - dim.positiv_saturated_femur[0], 0,
                    result.z - dim.positiv_saturated_femur[1]) <= dim.tibia_length;

        float deported_center_positiv[2];
        sincosf(dim.tibia_absolute_pos, &sin_angle_cox, &cos_angle_cox);
        deported_center_positiv[0] = dim.tibia_length * cos_angle_cox;
        deported_center_positiv[1] = dim.tibia_length * sin_angle_cox;
        // need it inside
        in_positiv_tib_circle =
            norm3df(result.x - deported_center_positiv[0], 0,
                    result.z - deported_center_positiv[1]) <= dim.femur_length;
    }

    bool reachability;
    reachability = (((!in_negativ_tib_circle) && (!in_positive_sat_circle)) &&
                    (in_negative_sat_circle || inside_femur || in_positiv_tib_circle));
    return reachability;
}

__device__ __forceinline__ bool multi_circle_validate(float x, float y, Circle* circleArr,
                                                      int number_of_circles) {
    for (int i = 0; i < number_of_circles; i++) {
        Circle& circle = circleArr[i];
        float distance;
        bool is_valid;

        distance_to_circumf(circle, x, y, distance, is_valid);
        if (not is_valid) {
            return false;
        }
    }
    return true; // return true is all valid
}

__device__ __forceinline__ bool multi_circle_clamp(float& x, float& y, Circle* circleArr,
                                                   int number_of_circles) {
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
        if (abs(circle.radius) > CIRCLE_MARGIN) {
            clamp_is_valid =
                multi_circle_validate(x_onthe_circle, y_onthe_circle, circleArr,
                                      number_of_circles - MAX_INTERSECT);
            overall_validity = overall_validity && validity;
        } else {
            clamp_is_valid = true;
        }
        // clamp_is_valid = true;
        bool closer_boundary = abs(previous_distance) > abs(distance);
        // closer_boundary = true;

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

__device__ __forceinline__ uchar find_region(float& x, float& z,
                                             const LegDimensions& dim) {
    float femur_angle_raw = atan2f(z, x);
    bool lower_region = (femur_angle_raw <= dim.min_angle_femur);
    bool upper_region = (femur_angle_raw >= dim.tibia_absolute_pos);
    uchar region;
    if (upper_region) {
        region = UPPER_REGION;
    } else if (lower_region) {
        region = LOWER_REGION;
    } else {
        region = MIDDLE_REGION;
    }
    return region;
}
template <int UseCase = REACH_USECASE>
__device__ __forceinline__ bool eval_plane_circles(float& x, float& y,
                                                   const LegDimensions& dim) {
    // Femur as the frame of reference witout rotation
    x -= dim.coxa_length;

    uchar region = find_region(x, y, dim);

    bool is_valid;
    if constexpr (UseCase == REACH_USECASE) {
        Circle circle_list[MAX_CIRCLES];
        auto* tail = circle_list;
        tail = insert_circles(dim, region, tail);
        // uchar number_of_circles = tail - head;
        const uchar number_of_circles = MAX_CIRCLES;

        is_valid = multi_circle_validate(x, y, circle_list, number_of_circles);
    } else if (UseCase == DIST_USECASE) {
        Circle circle_list[MAX_CIRCLE_INTER];
        auto* tail = circle_list;
        tail = insert_circles(dim, region, tail);
        tail = insert_intersec(dim, region, tail);
        // uchar number_of_circles = tail - circle_list;
        const uchar number_of_circles = MAX_CIRCLE_INTER;

        is_valid = multi_circle_clamp(x, y, circle_list, number_of_circles);
    } else {
        static_assert(UseCase == REACH_USECASE or UseCase == DIST_USECASE, "bad usecase");
    }

    return is_valid;
}

__device__ __forceinline__ bool reachability_circles(const float3& point,
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
        bool flip_flag = -signbit(result.x);

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

    if ((required_angle_coxa > dim.max_angle_coxa) ||
        (required_angle_coxa < dim.min_angle_coxa)) {
        return false;
    }

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox;
    float sin_angle_cox;
    cancel_coxa_rotation(result, required_angle_coxa, cos_angle_cox, sin_angle_cox);

    bool reachability = eval_plane_circles<REACH_USECASE>(result.x, result.z, dim);
    return reachability;
}

__device__ __forceinline__ bool place_on_vert_plane_circle(float& x, float& z,
                                                           const LegDimensions& dim) {
    return eval_plane_circles<DIST_USECASE>(x, z, dim);
}

__device__ __forceinline__ float3 distance_circles(const float3& point,
                                                   const LegDimensions& dim) {
    float3 closest = point;
    place_over_coxa(closest, dim);
    float3 closest_flip = closest;

    float coxangle = find_coxa_angle(closest);
    float coxangle_flip = (coxangle > 0) ? coxangle - pIgpu : coxangle + pIgpu;

    auto res =
        finish_finding_closest<bool, place_on_vert_plane_circle>(closest, dim, coxangle);
    auto resflip = finish_finding_closest<bool, place_on_vert_plane_circle>(
        closest_flip, dim, coxangle_flip);

    bool use_direct = norm3df(closest.x, closest.y, closest.z) <
                      norm3df(closest_flip.x, closest_flip.y, closest_flip.z);
    // use_direct = true;
    float3* result_to_use = (use_direct) ? &closest : &closest_flip;
    // if (!res) {
    // (*result_to_use).x /= 10;
    // (*result_to_use).y /= 10;
    // (*result_to_use).z /= 10;
    // }

    return *result_to_use;
}

__global__ void dist_kernel(const Array<float3> input, const LegDimensions dimensions,
                            Array<float3> const output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = dist_double_solf3(input.elements[i], dimensions);
        // output.elements[i] = input.elements[i];
    }
}

__global__ void reachability_kernel(const Array<float3> input,
                                    const LegDimensions dimensions,
                                    Array<bool> const output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = reachability_vect(input.elements[i], dimensions);
    }
}

__global__ void reachability_abs_tib_kernel(const Array<float3> input,
                                            const LegDimensions dimensions,
                                            Array<bool> const output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] =
            reachability_absolute_tibia_limit(input.elements[i], dimensions);
    }
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
        output.elements[i] = distance_circles(input.elements[i], dim);
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
