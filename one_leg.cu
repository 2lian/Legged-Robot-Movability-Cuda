#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "one_leg.cu.h"

__device__ void place_over_coxa(float3& coordinates, const LegDimensions& dim) {
    // Coxa as the frame of reference without rotation
    coordinates.x -= dim.body;
}

__device__ float find_coxa_angle(const float3& coordinates) {
    // finding coxa angle
    return atan2f(coordinates.y, coordinates.x);
}

__device__ void place_on_vert_plane(float& x, float& z,
                                    const LegDimensions& dim) {
    // Femur as the frame of reference witout rotation
    x -= dim.coxa_length;

    // finding femur angle
    float required_angle_femur = atan2f(z, x);
    float angle_overshoot = required_angle_femur;

    float overmargin = dim.femur_overmargin;

    // saturating femur angle for dist
    required_angle_femur = fmaxf(
        fminf(required_angle_femur, dim.max_angle_femur_w_margin + overmargin),
        dim.min_angle_femur_w_margin - overmargin);

    angle_overshoot = abs(angle_overshoot - fmaxf(fminf(required_angle_femur,
                                                        dim.max_angle_femur),
                                                  dim.min_angle_femur));

    // canceling femur rotation for dist
    float cos_angle_fem;
    float sin_angle_fem;
    sincosf(required_angle_femur, &sin_angle_fem, &cos_angle_fem);

    float center_adjust_factor = dim.middle_TG_radius *
                                 fmax(angle_overshoot / overmargin, 0.f) *
                                 sinf(fmax(angle_overshoot / overmargin, 0.f));

    // middle_TG as the frame of reference
    x -= (dim.middle_TG - center_adjust_factor) * cos_angle_fem;
    z -= (dim.middle_TG - center_adjust_factor) * sin_angle_fem;

    float zeroing_radius =
        fmax(dim.middle_TG_radius_w_margin - center_adjust_factor, 0.f);

    float magnitude = fmax(norm3df(x, z, 0.f), zeroing_radius);

    x -= zeroing_radius * x / magnitude;
    z -= zeroing_radius * z / magnitude;
}

__device__ void cancel_coxa_rotation(float3& coordinates,
                                     const float& coxa_angle,
                                     float& cosine_coxa_memory,
                                     float& sin_coxa_memory) {
    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    sincosf(-coxa_angle, &sin_coxa_memory, &cosine_coxa_memory);
    float buffer = coordinates.x * sin_coxa_memory;
    coordinates.x =
        coordinates.x * cosine_coxa_memory - coordinates.y * sin_coxa_memory;
    coordinates.y = buffer + coordinates.y * cosine_coxa_memory;
}

__device__ void restore_coxa_rotation(float3& coordinates,
                                      float& cosine_coxa_memory,
                                      float& sin_coxa_memory) {

    float buffer = coordinates.y * sin_coxa_memory;
    coordinates.y =
        -coordinates.x * sin_coxa_memory + coordinates.y * cosine_coxa_memory;
    coordinates.x = coordinates.x * cosine_coxa_memory + buffer;
}

__device__ void finish_finding_closest(float3& coordinates,
                                       const LegDimensions& dim,
                                       const float& coxa_angle) {
    // saturating coxa angle for dist
    float saturated_coxa_angle =
        fmaxf(fminf(coxa_angle, dim.max_angle_coxa_w_margin),
              dim.min_angle_coxa_w_margin);

    float cosine_coxa_memory;
    float sin_coxa_memory;
    cancel_coxa_rotation(coordinates, saturated_coxa_angle, cosine_coxa_memory,
                         sin_coxa_memory);

    place_on_vert_plane(coordinates.x, coordinates.z, dim);

    restore_coxa_rotation(coordinates, cosine_coxa_memory, sin_coxa_memory);
}

__device__ float3 dist_double_solf3(const float3& point,
                                    const LegDimensions& dim) {
    float3 closest = point;
    place_over_coxa(closest, dim);
    float3 closest_flip = closest;

    float coxangle = find_coxa_angle(closest);
    float coxangle_flip = (coxangle > 0) ? coxangle - pIgpu : coxangle + pIgpu;

    finish_finding_closest(closest, dim, coxangle);
    finish_finding_closest(closest_flip, dim, (coxangle_flip));

    float3* result_to_use =
        (norm3df(closest.x, closest.y, closest.z) <
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
    bool flip_flag = signbit(result.x);

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
    cancel_coxa_rotation(result, required_angle_coxa, cos_angle_cox,
                         sin_angle_cox);

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

__global__ void dist_kernel(const Array<float3> input,
                            const LegDimensions dimensions,
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

__device__ float3 forward_kinematics(const float coxa, const float femur,
                                     const float tibia,
                                     const LegDimensions dim) {
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
        output_xyz.elements[i] = forward_kinematics(
            angles_3_input.elements[i].x, angles_3_input.elements[i].y,
            angles_3_input.elements[i].z, dim);
    }
}
