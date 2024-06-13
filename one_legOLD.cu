#include "HeaderCPP.h"
#include "HeaderCUDA.h"
// #include "circles.cu.h"
// #include "one_leg.cu.h"

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

