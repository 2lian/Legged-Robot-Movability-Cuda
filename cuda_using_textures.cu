#include <iostream>
#include <fstream>
#include "Header.h"

RobotDimensions dim_of_SCARE() {
    struct RobotDimensions scare{};

    scare.pI = 3.141592653589793238462643383279502884197169f;
    scare.body = 185.0f;
    scare.coxa_angle_deg = 60.0f;
    scare.coxa_length = 165.0f;
    scare.tibia_angle_deg = 120.0f; //90
    scare.tibia_length = 190.0f;
    scare.tibia_length_squared = scare.tibia_length * scare.tibia_length;
    scare.femur_angle_deg = 120.0f; //120
    scare.femur_length = 200.0f; //200

    scare.max_angle_coxa = scare.pI / 180.0f * scare.coxa_angle_deg;
    scare.min_angle_coxa = -scare.pI / 180.0f * scare.coxa_angle_deg;

    scare.max_angle_coxa_w_margin = scare.pI / 180.0f * (scare.coxa_angle_deg - 5.0f);
    scare.min_angle_coxa_w_margin = -scare.pI / 180.0f * (scare.coxa_angle_deg - 5.0f);

    scare.max_angle_tibia = scare.pI / 180.0f * scare.tibia_angle_deg;
    scare.min_angle_tibia = -scare.pI / 180.0f * scare.tibia_angle_deg;

    scare.max_angle_femur = scare.pI / 180.0f * scare.femur_angle_deg;
    scare.min_angle_femur = -scare.pI / 180.0f * scare.femur_angle_deg;

    scare.max_angle_femur_w_margin = scare.pI / 180.0f * (scare.femur_angle_deg + 0.0f);
    scare.min_angle_femur_w_margin = -scare.pI / 180.0f * (scare.femur_angle_deg + 0.0f);
    scare.max_tibia_to_gripper_dist = scare.tibia_length + scare.femur_length;

    scare.positiv_saturated_femur[0] = cos(scare.max_angle_femur) * scare.tibia_length;
    scare.positiv_saturated_femur[1] = sin(scare.max_angle_femur) * scare.tibia_length;

    scare.negativ_saturated_femur[0] = cos(scare.min_angle_femur) * scare.tibia_length;
    scare.negativ_saturated_femur[1] = sin(scare.min_angle_femur) * scare.tibia_length;

    scare.fem_tib_min_host[0] = scare.tibia_length + scare.femur_length * cos(scare.pI / 180.0f * scare.tibia_angle_deg);
    scare.fem_tib_min_host[1] = scare.femur_length * sin(scare.pI / 180.0f * scare.tibia_angle_deg);

    scare.min_tibia_to_gripper_dist = sqrt(scare.fem_tib_min_host[0] * scare.fem_tib_min_host[0]
                                           + scare.fem_tib_min_host[1] * scare.fem_tib_min_host[1]);
    scare.middle_TG = (scare.max_tibia_to_gripper_dist + scare.min_tibia_to_gripper_dist) / 2.0f;
    scare.middle_TG_radius = (scare.max_tibia_to_gripper_dist - scare.min_tibia_to_gripper_dist)/2;
    scare.middle_TG_radius_w_margin = scare.middle_TG_radius-10.f;

    scare.femur_overmargin = -scare.max_angle_femur + atan2f(
            scare.femur_length * sinf(scare.max_angle_femur) +
            (scare.tibia_length - 10) * sinf(scare.max_angle_femur + scare.max_angle_tibia)
            ,
            scare.femur_length * cosf(scare.max_angle_femur) +
            (scare.tibia_length - 10) * cosf(scare.max_angle_femur + scare.max_angle_tibia)
            );

    return scare;
}


__constant__ __device__ float VirdisColorMapData[256][3] =
        {
                { 0.f, 0.f, 0.f },
//                { 0.267004, 0.004874, 0.329415 },
                { 0.268510, 0.009605, 0.335427 },
                { 0.269944, 0.014625, 0.341379 },
                { 0.271305, 0.019942, 0.347269 },
                { 0.272594, 0.025563, 0.353093 },
                { 0.273809, 0.031497, 0.358853 },
                { 0.274952, 0.037752, 0.364543 },
                { 0.276022, 0.044167, 0.370164 },
                { 0.277018, 0.050344, 0.375715 },
                { 0.277941, 0.056324, 0.381191 },
                { 0.278791, 0.062145, 0.386592 },
                { 0.279566, 0.067836, 0.391917 },
                { 0.280267, 0.073417, 0.397163 },
                { 0.280894, 0.078907, 0.402329 },
                { 0.281446, 0.084320, 0.407414 },
                { 0.281924, 0.089666, 0.412415 },
                { 0.282327, 0.094955, 0.417331 },
                { 0.282656, 0.100196, 0.422160 },
                { 0.282910, 0.105393, 0.426902 },
                { 0.283091, 0.110553, 0.431554 },
                { 0.283197, 0.115680, 0.436115 },
                { 0.283229, 0.120777, 0.440584 },
                { 0.283187, 0.125848, 0.444960 },
                { 0.283072, 0.130895, 0.449241 },
                { 0.282884, 0.135920, 0.453427 },
                { 0.282623, 0.140926, 0.457517 },
                { 0.282290, 0.145912, 0.461510 },
                { 0.281887, 0.150881, 0.465405 },
                { 0.281412, 0.155834, 0.469201 },
                { 0.280868, 0.160771, 0.472899 },
                { 0.280255, 0.165693, 0.476498 },
                { 0.279574, 0.170599, 0.479997 },
                { 0.278826, 0.175490, 0.483397 },
                { 0.278012, 0.180367, 0.486697 },
                { 0.277134, 0.185228, 0.489898 },
                { 0.276194, 0.190074, 0.493001 },
                { 0.275191, 0.194905, 0.496005 },
                { 0.274128, 0.199721, 0.498911 },
                { 0.273006, 0.204520, 0.501721 },
                { 0.271828, 0.209303, 0.504434 },
                { 0.270595, 0.214069, 0.507052 },
                { 0.269308, 0.218818, 0.509577 },
                { 0.267968, 0.223549, 0.512008 },
                { 0.266580, 0.228262, 0.514349 },
                { 0.265145, 0.232956, 0.516599 },
                { 0.263663, 0.237631, 0.518762 },
                { 0.262138, 0.242286, 0.520837 },
                { 0.260571, 0.246922, 0.522828 },
                { 0.258965, 0.251537, 0.524736 },
                { 0.257322, 0.256130, 0.526563 },
                { 0.255645, 0.260703, 0.528312 },
                { 0.253935, 0.265254, 0.529983 },
                { 0.252194, 0.269783, 0.531579 },
                { 0.250425, 0.274290, 0.533103 },
                { 0.248629, 0.278775, 0.534556 },
                { 0.246811, 0.283237, 0.535941 },
                { 0.244972, 0.287675, 0.537260 },
                { 0.243113, 0.292092, 0.538516 },
                { 0.241237, 0.296485, 0.539709 },
                { 0.239346, 0.300855, 0.540844 },
                { 0.237441, 0.305202, 0.541921 },
                { 0.235526, 0.309527, 0.542944 },
                { 0.233603, 0.313828, 0.543914 },
                { 0.231674, 0.318106, 0.544834 },
                { 0.229739, 0.322361, 0.545706 },
                { 0.227802, 0.326594, 0.546532 },
                { 0.225863, 0.330805, 0.547314 },
                { 0.223925, 0.334994, 0.548053 },
                { 0.221989, 0.339161, 0.548752 },
                { 0.220057, 0.343307, 0.549413 },
                { 0.218130, 0.347432, 0.550038 },
                { 0.216210, 0.351535, 0.550627 },
                { 0.214298, 0.355619, 0.551184 },
                { 0.212395, 0.359683, 0.551710 },
                { 0.210503, 0.363727, 0.552206 },
                { 0.208623, 0.367752, 0.552675 },
                { 0.206756, 0.371758, 0.553117 },
                { 0.204903, 0.375746, 0.553533 },
                { 0.203063, 0.379716, 0.553925 },
                { 0.201239, 0.383670, 0.554294 },
                { 0.199430, 0.387607, 0.554642 },
                { 0.197636, 0.391528, 0.554969 },
                { 0.195860, 0.395433, 0.555276 },
                { 0.194100, 0.399323, 0.555565 },
                { 0.192357, 0.403199, 0.555836 },
                { 0.190631, 0.407061, 0.556089 },
                { 0.188923, 0.410910, 0.556326 },
                { 0.187231, 0.414746, 0.556547 },
                { 0.185556, 0.418570, 0.556753 },
                { 0.183898, 0.422383, 0.556944 },
                { 0.182256, 0.426184, 0.557120 },
                { 0.180629, 0.429975, 0.557282 },
                { 0.179019, 0.433756, 0.557430 },
                { 0.177423, 0.437527, 0.557565 },
                { 0.175841, 0.441290, 0.557685 },
                { 0.174274, 0.445044, 0.557792 },
                { 0.172719, 0.448791, 0.557885 },
                { 0.171176, 0.452530, 0.557965 },
                { 0.169646, 0.456262, 0.558030 },
                { 0.168126, 0.459988, 0.558082 },
                { 0.166617, 0.463708, 0.558119 },
                { 0.165117, 0.467423, 0.558141 },
                { 0.163625, 0.471133, 0.558148 },
                { 0.162142, 0.474838, 0.558140 },
                { 0.160665, 0.478540, 0.558115 },
                { 0.159194, 0.482237, 0.558073 },
                { 0.157729, 0.485932, 0.558013 },
                { 0.156270, 0.489624, 0.557936 },
                { 0.154815, 0.493313, 0.557840 },
                { 0.153364, 0.497000, 0.557724 },
                { 0.151918, 0.500685, 0.557587 },
                { 0.150476, 0.504369, 0.557430 },
                { 0.149039, 0.508051, 0.557250 },
                { 0.147607, 0.511733, 0.557049 },
                { 0.146180, 0.515413, 0.556823 },
                { 0.144759, 0.519093, 0.556572 },
                { 0.143343, 0.522773, 0.556295 },
                { 0.141935, 0.526453, 0.555991 },
                { 0.140536, 0.530132, 0.555659 },
                { 0.139147, 0.533812, 0.555298 },
                { 0.137770, 0.537492, 0.554906 },
                { 0.136408, 0.541173, 0.554483 },
                { 0.135066, 0.544853, 0.554029 },
                { 0.133743, 0.548535, 0.553541 },
                { 0.132444, 0.552216, 0.553018 },
                { 0.131172, 0.555899, 0.552459 },
                { 0.129933, 0.559582, 0.551864 },
                { 0.128729, 0.563265, 0.551229 },
                { 0.127568, 0.566949, 0.550556 },
                { 0.126453, 0.570633, 0.549841 },
                { 0.125394, 0.574318, 0.549086 },
                { 0.124395, 0.578002, 0.548287 },
                { 0.123463, 0.581687, 0.547445 },
                { 0.122606, 0.585371, 0.546557 },
                { 0.121831, 0.589055, 0.545623 },
                { 0.121148, 0.592739, 0.544641 },
                { 0.120565, 0.596422, 0.543611 },
                { 0.120092, 0.600104, 0.542530 },
                { 0.119738, 0.603785, 0.541400 },
                { 0.119512, 0.607464, 0.540218 },
                { 0.119423, 0.611141, 0.538982 },
                { 0.119483, 0.614817, 0.537692 },
                { 0.119699, 0.618490, 0.536347 },
                { 0.120081, 0.622161, 0.534946 },
                { 0.120638, 0.625828, 0.533488 },
                { 0.121380, 0.629492, 0.531973 },
                { 0.122312, 0.633153, 0.530398 },
                { 0.123444, 0.636809, 0.528763 },
                { 0.124780, 0.640461, 0.527068 },
                { 0.126326, 0.644107, 0.525311 },
                { 0.128087, 0.647749, 0.523491 },
                { 0.130067, 0.651384, 0.521608 },
                { 0.132268, 0.655014, 0.519661 },
                { 0.134692, 0.658636, 0.517649 },
                { 0.137339, 0.662252, 0.515571 },
                { 0.140210, 0.665859, 0.513427 },
                { 0.143303, 0.669459, 0.511215 },
                { 0.146616, 0.673050, 0.508936 },
                { 0.150148, 0.676631, 0.506589 },
                { 0.153894, 0.680203, 0.504172 },
                { 0.157851, 0.683765, 0.501686 },
                { 0.162016, 0.687316, 0.499129 },
                { 0.166383, 0.690856, 0.496502 },
                { 0.170948, 0.694384, 0.493803 },
                { 0.175707, 0.697900, 0.491033 },
                { 0.180653, 0.701402, 0.488189 },
                { 0.185783, 0.704891, 0.485273 },
                { 0.191090, 0.708366, 0.482284 },
                { 0.196571, 0.711827, 0.479221 },
                { 0.202219, 0.715272, 0.476084 },
                { 0.208030, 0.718701, 0.472873 },
                { 0.214000, 0.722114, 0.469588 },
                { 0.220124, 0.725509, 0.466226 },
                { 0.226397, 0.728888, 0.462789 },
                { 0.232815, 0.732247, 0.459277 },
                { 0.239374, 0.735588, 0.455688 },
                { 0.246070, 0.738910, 0.452024 },
                { 0.252899, 0.742211, 0.448284 },
                { 0.259857, 0.745492, 0.444467 },
                { 0.266941, 0.748751, 0.440573 },
                { 0.274149, 0.751988, 0.436601 },
                { 0.281477, 0.755203, 0.432552 },
                { 0.288921, 0.758394, 0.428426 },
                { 0.296479, 0.761561, 0.424223 },
                { 0.304148, 0.764704, 0.419943 },
                { 0.311925, 0.767822, 0.415586 },
                { 0.319809, 0.770914, 0.411152 },
                { 0.327796, 0.773980, 0.406640 },
                { 0.335885, 0.777018, 0.402049 },
                { 0.344074, 0.780029, 0.397381 },
                { 0.352360, 0.783011, 0.392636 },
                { 0.360741, 0.785964, 0.387814 },
                { 0.369214, 0.788888, 0.382914 },
                { 0.377779, 0.791781, 0.377939 },
                { 0.386433, 0.794644, 0.372886 },
                { 0.395174, 0.797475, 0.367757 },
                { 0.404001, 0.800275, 0.362552 },
                { 0.412913, 0.803041, 0.357269 },
                { 0.421908, 0.805774, 0.351910 },
                { 0.430983, 0.808473, 0.346476 },
                { 0.440137, 0.811138, 0.340967 },
                { 0.449368, 0.813768, 0.335384 },
                { 0.458674, 0.816363, 0.329727 },
                { 0.468053, 0.818921, 0.323998 },
                { 0.477504, 0.821444, 0.318195 },
                { 0.487026, 0.823929, 0.312321 },
                { 0.496615, 0.826376, 0.306377 },
                { 0.506271, 0.828786, 0.300362 },
                { 0.515992, 0.831158, 0.294279 },
                { 0.525776, 0.833491, 0.288127 },
                { 0.535621, 0.835785, 0.281908 },
                { 0.545524, 0.838039, 0.275626 },
                { 0.555484, 0.840254, 0.269281 },
                { 0.565498, 0.842430, 0.262877 },
                { 0.575563, 0.844566, 0.256415 },
                { 0.585678, 0.846661, 0.249897 },
                { 0.595839, 0.848717, 0.243329 },
                { 0.606045, 0.850733, 0.236712 },
                { 0.616293, 0.852709, 0.230052 },
                { 0.626579, 0.854645, 0.223353 },
                { 0.636902, 0.856542, 0.216620 },
                { 0.647257, 0.858400, 0.209861 },
                { 0.657642, 0.860219, 0.203082 },
                { 0.668054, 0.861999, 0.196293 },
                { 0.678489, 0.863742, 0.189503 },
                { 0.688944, 0.865448, 0.182725 },
                { 0.699415, 0.867117, 0.175971 },
                { 0.709898, 0.868751, 0.169257 },
                { 0.720391, 0.870350, 0.162603 },
                { 0.730889, 0.871916, 0.156029 },
                { 0.741388, 0.873449, 0.149561 },
                { 0.751884, 0.874951, 0.143228 },
                { 0.762373, 0.876424, 0.137064 },
                { 0.772852, 0.877868, 0.131109 },
                { 0.783315, 0.879285, 0.125405 },
                { 0.793760, 0.880678, 0.120005 },
                { 0.804182, 0.882046, 0.114965 },
                { 0.814576, 0.883393, 0.110347 },
                { 0.824940, 0.884720, 0.106217 },
                { 0.835270, 0.886029, 0.102646 },
                { 0.845561, 0.887322, 0.099702 },
                { 0.855810, 0.888601, 0.097452 },
                { 0.866013, 0.889868, 0.095953 },
                { 0.876168, 0.891125, 0.095250 },
                { 0.886271, 0.892374, 0.095374 },
                { 0.896320, 0.893616, 0.096335 },
                { 0.906311, 0.894855, 0.098125 },
                { 0.916242, 0.896091, 0.100717 },
                { 0.926106, 0.897330, 0.104071 },
                { 0.935904, 0.898570, 0.108131 },
                { 0.945636, 0.899815, 0.112838 },
                { 0.955300, 0.901065, 0.118128 },
                { 0.964894, 0.902323, 0.123941 },
                { 0.974417, 0.903590, 0.130215 },
                { 0.983868, 0.904867, 0.136897 },
                { 0.993248, 0.906157, 0.143936 }
        };


__global__
void empty_kernel() {
}


__device__ float sumOfSquares3df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
}
__device__ float sumOfSquares2df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1];
}

__device__
float place_over_coxa(float3& coordinates, RobotDimensions& dim)
{
    // Coxa as the frame of reference without rotation
    coordinates.x -= dim.body;
}

__device__
float find_coxa_angle(float3& coordinates)
{
    // finding coxa angle
    return atan2f(coordinates.y, coordinates.x);
}

__device__
void find_vert_plane_dist(float& x, float& z, RobotDimensions& dim)
{
    // Femur as the frame of reference witout rotation
    x -= dim.coxa_length;

    // finding femur angle
    float required_angle_femur = atan2f(z, x);
    float angle_overshoot = required_angle_femur;

    float overmargin = dim.femur_overmargin;

    // saturating femur angle for dist
    required_angle_femur = fmaxf(fminf(required_angle_femur,
                                       dim.max_angle_femur_w_margin + overmargin),
                                 dim.min_angle_femur_w_margin - overmargin);

    angle_overshoot = abs(angle_overshoot - fmaxf(fminf(
            required_angle_femur,
                                                        dim.max_angle_femur),
                                                  dim.min_angle_femur));

    // canceling femur rotation for dist
    float cos_angle_fem;
    float sin_angle_fem;
    sincosf(required_angle_femur, &sin_angle_fem, &cos_angle_fem);

    float center_adjust_factor = dim.middle_TG_radius *
                                 fmax(angle_overshoot/overmargin,0.f) *
                                 sinf(fmax(angle_overshoot/overmargin,0.f))
            ;

    // middle_TG as the frame of reference
    x -= (dim.middle_TG - center_adjust_factor) * cos_angle_fem;
    z -= (dim.middle_TG - center_adjust_factor) * sin_angle_fem;

    float zeroing_radius = fmax(dim.middle_TG_radius_w_margin - center_adjust_factor, 0.f);

    float magnitude = fmax(
            norm3df(x, z, 0.f),
            zeroing_radius
            );

    x -= zeroing_radius*x/magnitude;
    z -= zeroing_radius*z/magnitude;
}

__device__
void finish_finding_dist(float3& coordinates, RobotDimensions& dim, const float coxa_angle)
{
    // saturating coxa angle for dist
    float test = fmax(norm3df(coordinates.x, coordinates.y, 0.f), 0.f);
    float sub = fmax(dim.max_angle_coxa - asin(10.f/ test), 0.f);
    float saturated_coxa_angle = fmaxf(fminf(coxa_angle, sub), -sub);
//    float saturated_coxa_angle = fmaxf(fminf(coxa_angle, dim.max_angle_coxa), dim.min_angle_coxa);

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox;
    float sin_angle_cox;
    sincosf(-saturated_coxa_angle, &sin_angle_cox, &cos_angle_cox);
    float buffer = coordinates.x * sin_angle_cox;
    coordinates.x = coordinates.x * cos_angle_cox - coordinates.y * sin_angle_cox;
    coordinates.y = buffer + coordinates.y * cos_angle_cox;

    find_vert_plane_dist(coordinates.x, coordinates.z, dim);

    buffer = coordinates.y * sin_angle_cox;
    coordinates.y = -coordinates.x * sin_angle_cox + coordinates.y * cos_angle_cox;
    coordinates.x = coordinates.x * cos_angle_cox + buffer;
}

__device__
float3 dist_double_solf3(float3 point, RobotDimensions& dim)
//
{
    float3 coordinate_noflip = point;
    place_over_coxa(coordinate_noflip, dim);
    float3 coordinate_flip = coordinate_noflip;


//    coordinate_noflip.x -= 10;
    float coxangle = find_coxa_angle(coordinate_noflip);

    finish_finding_dist(coordinate_noflip,
                        dim,
                        coxangle);
    finish_finding_dist(coordinate_flip,
                        dim,
                        ((coxangle > 0)? coxangle - dim.pI : coxangle + dim.pI));

    float3* result_to_use = (
            norm3df(coordinate_noflip.x, coordinate_noflip.y, coordinate_noflip.z)
            <
            norm3df(coordinate_flip.x, coordinate_flip.y, coordinate_flip.z)
                            ) ? &coordinate_noflip : &coordinate_flip;

    // result_to_use = result_flip;
    return *result_to_use;

}

__device__
bool reachability_vect(const float3& point, RobotDimensions& dim)
// no angle flipping
{
    // Coxa as the frame of reference without rotation
    float3 result;
    result = point;
    result.x -= dim.body;

    // finding coxa angle
    float required_angle_coxa = atan2f(result.y, result.x);

    // flipping angle if above +-90deg
    required_angle_coxa = fmodf(required_angle_coxa + dim.pI / 2.f + 2.f * dim.pI, dim.pI) - dim.pI / 2.f;

    if ((required_angle_coxa > dim.max_angle_coxa) || (required_angle_coxa < dim.min_angle_coxa)){
        return false;
    }

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox = cosf(-required_angle_coxa);
    float sin_angle_cox = sinf(-required_angle_coxa);
    float buffer = result.x * sin_angle_cox;
    result.x = result.x * cos_angle_cox - result.y * sin_angle_cox;
    result.y = buffer + result.y * cos_angle_cox;

    // Femur as the frame of reference witout rotation
    result.x -= dim.coxa_length;

    float linnorm = norm3df(result.x, result.y, result.z);

    if ((linnorm < dim.min_tibia_to_gripper_dist) || (linnorm > dim.max_tibia_to_gripper_dist)){
        return false;
    }

    // finding femur angle
    float required_angle_femur = atan2f(result.z, result.x);

    if ((required_angle_femur > dim.min_angle_femur) && (required_angle_femur < dim.max_angle_femur)) {
        return true;
    }

    linnorm = fminf(
            norm3df(result.x - dim.positiv_saturated_femur[0], 0, result.z - dim.positiv_saturated_femur[1])
            ,
            norm3df(result.x - dim.negativ_saturated_femur[0], 0, result.z - dim.negativ_saturated_femur[1])
    );

    return linnorm < dim.femur_length;
}

__global__ void toVirdisUint_kernel(Matrixf table, unsigned char* pixels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < table.height; i += stride) {
        const int x = (int) floorf(
                fmaxf(
                        fminf(255.f, table.elements[i]),
                        0.f)
        );
        const float color[3] = {VirdisColorMapData[x][0], VirdisColorMapData[x][1], VirdisColorMapData[x][2]};
        pixels[i * 4 + 0] = (unsigned char) floorf(color[0]*255.f);
        pixels[i * 4 + 1] = (unsigned char) floorf(color[1]*255.f);
        pixels[i * 4 + 2] = (unsigned char) floorf(color[2]*255.f);
        pixels[i * 4 + 3] = (unsigned char) (1*255);
    }
}

__global__
void switch_zy_kernel(Matrixf table)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        float buffer = table.elements[i * table.width + 1];
        table.elements[i * table.width + 1] = table.elements[i * table.width + 2];
        table.elements[i * table.width + 2] = buffer;
    }
}

__global__
void switch_zy_kernelf3(Arrayf3 arr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arr.length; i += stride) {
        const float buffer = arr.elements[i].y;
        arr.elements[i].y = arr.elements[i].z;
        arr.elements[i].z = buffer;
    }
}

__global__
void norm3df_kernel(Matrixf table, Matrixf result_table)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        result_table.elements[i] = norm3df(table.elements[i * table.width],
                                           table.elements[i * table.width+1],
                                           table.elements[i * table.width+2]);
    }
}

__global__
void change_z_kernel(Matrixf table, float zval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        table.elements[i * table.width+2] = zval;
    }
}

__global__
void change_y_kernel(Matrixf table, float zval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        table.elements[i * table.width+1] = zval;
    }
}

__global__
void change_z_kernelf3(Arrayf3 arr, float zval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arr.length; i += stride) {
        arr.elements[i].z = zval;
    }
}

__global__
void change_y_kernelf3(Arrayf3 arr, float zval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arr.length; i += stride) {
        arr.elements[i].y = zval;
    }
}

__global__
void dist2virdis_pipelinef3(Arrayf3 arr, RobotDimensions dimensions, unsigned char* pixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arr.length; i += stride) {

        // dist func
        float3 result = dist_double_solf3(arr.elements[i], dimensions);

        //norm func
        // norm stored in result[0]
        result.x = norm3df(result.x,
                            result.y,
                            result.z)
                    /2;

        // virdis colormaping

        const int x = (int)
                floorf(
                        fmaxf(
                                fminf(255.f, result.x + 0.95f),
                                0.f)
                );
        const float color[3] = {VirdisColorMapData[x][0], VirdisColorMapData[x][1], VirdisColorMapData[x][2]};

        float rBelow = floorf(color[0] * 255.f);
        float gBelow = floorf(color[1] * 255.f);
        float bBelow = floorf(color[2] * 255.f);
        float aBelow = 255;  // This is the new alpha value

        // Old pixel values
        auto rAbove = (float)pixels[i * 4 + 0];
        auto gAbove = (float)pixels[i * 4 + 1];
        auto bAbove = (float)pixels[i * 4 + 2];
        auto aAbove = (float)pixels[i * 4 + 3];

        // Calculate the new pixel values with alpha compositing
        float alphaAbove = aAbove / 255.0f;
        float alphaBelow = aBelow / 255.0f;
        float alphaResult = alphaAbove + alphaBelow * (1.0f - alphaAbove);

        auto rMerged = (unsigned char)((rAbove * alphaAbove + rBelow * alphaBelow * (1.0f - alphaAbove)) / alphaResult);
        auto gMerged = (unsigned char)((gAbove * alphaAbove + gBelow * alphaBelow * (1.0f - alphaAbove)) / alphaResult);
        auto bMerged = (unsigned char)((bAbove * alphaAbove + bBelow * alphaBelow * (1.0f - alphaAbove)) / alphaResult);
        auto aMerged = (unsigned char)(alphaResult * 255.0f);

        // Update the pixel values
        pixels[i * 4 + 0] = rMerged;
        pixels[i * 4 + 1] = gMerged;
        pixels[i * 4 + 2] = bMerged;
        pixels[i * 4 + 3] = aMerged;
    }
}

__global__
void reachability2img_vect_pipeline(Arrayf3 arr, RobotDimensions dimensions, unsigned char* pixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < arr.length; i += stride) {
        bool result = reachability_vect(arr.elements[i], dimensions);

        unsigned char val = (result)? 0 : 255;
        for (int n=0; n<4; n++){
            pixels[i * 4 + n] = val;
        }
        pixels[i * 4 + 3] = (unsigned char) (1*255);
    }
}

__global__ void gauss_derivate(const unsigned char* pixels, unsigned char* outputImage, int imageWidth, int imageHeight) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned char inv_gaussianKernel1[9] = {
            0, 8, 0,
            8, 2, 8,
            0, 8, 0,
    };
    unsigned char inv_gaussianKernel2[9] = {
            16, 8, 16,
            8, 4, 8,
            16, 8, 16,
    };

    for (int subindex = index; subindex < imageWidth * imageHeight; subindex += stride) {
        int centerX = subindex % imageWidth;
        int centerY = subindex / imageWidth;
        unsigned char sum1 = 0;
        unsigned char sum2 = 0;

        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int pixelX = centerX + i;
                int pixelY = centerY + j;
                int kernelIndex = (j + 1) * 3 + (i + 1);

                if (pixelX >= 0 && pixelX < imageWidth && pixelY >= 0 && pixelY < imageHeight) {
                    unsigned char pixelValue = pixels[(pixelY * imageWidth + pixelX)*4];
                    sum1 += pixelValue / inv_gaussianKernel1[kernelIndex];
                    sum2 += pixelValue / inv_gaussianKernel2[kernelIndex];
                }
            }
        }
        unsigned char d = sum1 - sum2;
        outputImage[subindex*4] = d;
        outputImage[subindex*4 + 3] = d;
    }
}

AutoEstimator::AutoEstimator(int pxWidth, int pxHeight, float scale) {
    blockSize = 1024;
    verbose = true;
    screenWidth = pxWidth;
    screenHeight = pxHeight;
    dimensions = dim_of_SCARE();
    rows = screenWidth * screenHeight;

    table_input.width = 3; table_input.height = rows;
    table_input.elements = new float[table_input.width * table_input.height];

    arr_input.length = rows;
    arr_input.elements = new float3[arr_input.length];

    result.width = 3; result.height = rows;
    result.elements = new float[result.width * result.height];

    result_norm.width = 1; result_norm.height = rows;
    result_norm.elements = new float[result_norm.width * result_norm.height];

    virdisTexture = new unsigned char [4 * rows];

    targetset.width = 3; targetset.height = 6;
    targetset.elements = new float[targetset.width * targetset.height];

    for (int leg=0; leg < 6; leg++) {
        targetset.elements[leg*3 + 0] = cos(3.14159f/3.f * (float)leg) * 420;
        targetset.elements[leg*3 + 1] = sin(3.14159f/3.f * (float)leg) * 420;
        targetset.elements[leg*3 + 2] = 0.f;
    }

    table_input_gpu.width = table_input.width; table_input_gpu.height = table_input.height;
    result_gpu.width = table_input_gpu.width; result_gpu.height = table_input_gpu.height;
    result_norm_gpu.width = result_norm.width; result_norm_gpu.height = table_input_gpu.height;
    targetset_gpu.width = 3; targetset_gpu.height = 6;

    arr_input_gpu.length = screenWidth * screenHeight;

    numBlocks = (rows + blockSize - 1) / blockSize;
    error_check();
    setup_kernel();
    input_as_grid(scale);
    allocate_gpu_mem();
    copy_input_cpu2gpu();
}

void AutoEstimator::error_check(){
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or exit the program as needed.
    }
}

void AutoEstimator::input_as_grid(const float scale_factor) const{
    for (int i = 0; i < screenHeight; i++) {
        for (int j = 0; j < screenWidth; j++) {
            int row = i * screenWidth + j;

            float x = (float)(j - screenWidth /2)*scale_factor;
            float y = -(float)(i - screenHeight / 2)*scale_factor;
            float z = 0.f * scale_factor;

            // X
            *(table_input.elements + row * table_input.width + 0)
                    //= -1000.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2000.0f;
                    = x;
            arr_input.elements[row].x = x;
            // Y
            *(table_input.elements + row * table_input.width + 1)
                    //= -1000.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2000.0f;
                    = y;
            arr_input.elements[row].y = y;
            // Z
            *(table_input.elements + row * table_input.width + 2)
                    //= -500.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f;
                    = z;
            arr_input.elements[row].z = z;
        }
    }
    if (verbose) { std::cout << "Host grid generated" << std::endl; }
}

void AutoEstimator::change_z_value(float value){

    change_z_kernel<<<numBlocks, blockSize >>>(table_input_gpu, value);
    change_z_kernelf3<<<numBlocks, blockSize >>>(arr_input_gpu, value);
    cudaDeviceSynchronize(); error_check();
}

void AutoEstimator::change_y_value(float value){

    change_y_kernel<<<numBlocks, blockSize >>>(table_input_gpu, value);
    change_y_kernelf3<<<numBlocks, blockSize >>>(arr_input_gpu, value);
    cudaDeviceSynchronize(); error_check();
}

void AutoEstimator::switch_zy(){

    switch_zy_kernel<<<numBlocks, blockSize >>>(table_input_gpu);
    switch_zy_kernelf3<<<numBlocks, blockSize >>>(arr_input_gpu);
    cudaDeviceSynchronize(); error_check();
}

void AutoEstimator::allocate_gpu_mem(){
    cudaMalloc(&table_input_gpu.elements, table_input_gpu.width * table_input_gpu.height * sizeof(float));
    cudaMalloc(&arr_input_gpu.elements, arr_input_gpu.length * sizeof(float3));

    cudaMalloc(&result_gpu.elements, result_gpu.width * result_gpu.height * sizeof(float));
    cudaMemset(result_gpu.elements, 0, rows * sizeof(int));

    cudaMalloc(&result_norm_gpu.elements, result_norm_gpu.width * result_norm_gpu.height * sizeof(float));

    cudaMalloc(&virdisTexture_gpu, 4 * rows * sizeof(unsigned char));

    cudaMalloc(&targetset_gpu.elements, targetset_gpu.width * targetset_gpu.height * sizeof(float));

    cudaMalloc(&gpu_accumulator, rows * sizeof(int));
    cudaMemset(gpu_accumulator, 0, rows * sizeof(int));

    if (verbose) { std::cout << "GPU memory allocated" << std::endl; }
    error_check();
}

void AutoEstimator::copy_input_cpu2gpu(){
    cudaMemcpy(table_input_gpu.elements,
               table_input.elements,
               table_input.width * table_input.height * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(arr_input_gpu.elements,
               arr_input.elements,
               arr_input.length * sizeof(float3),
               cudaMemcpyHostToDevice);

    cudaMemcpy(targetset_gpu.elements,
               targetset.elements,
               targetset.width * targetset.height * sizeof(float),
               cudaMemcpyHostToDevice);

    if (verbose) { std::cout << "Host data copied to GPU" << std::endl; }
    error_check();
}

void AutoEstimator::setup_kernel(){
    std::wcout << "Threads per block: " << blockSize << "\nNumber of blocks: " << numBlocks << std::endl;
    empty_kernel<<<numBlocks, blockSize >>>();
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Empty kernels started" << std::endl;}
    error_check();
}

void AutoEstimator::compute_result_norm(){
    if (verbose) { std::cout << "Compute norm started" << std::endl;}
    norm3df_kernel<<<numBlocks, blockSize >>>(result_gpu,
                                              result_norm_gpu);
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::derivate_output(){
    if (verbose) { std::cout << "Compute norm started" << std::endl;}
    unsigned char* buffer;
    cudaMalloc(&buffer, 4 * rows * sizeof(unsigned char));
    cudaMemset(buffer, 0, 4 * rows * sizeof(unsigned char));
    gauss_derivate<<<numBlocks, blockSize >>>(virdisTexture_gpu,
                                              buffer,
                                              screenWidth,
                                              screenHeight);
    cudaDeviceSynchronize();
    cudaFree(virdisTexture_gpu);
    virdisTexture_gpu = buffer;
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::reset_image(){
    if (verbose) { std::cout << "Compute norm started" << std::endl;}
    cudaMemset(virdisTexture_gpu, 0, 4 * rows * sizeof(unsigned char));
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::convert_to_virdis(){
    if (verbose) { std::cout << "Compute virdis started" << std::endl;}
    toVirdisUint_kernel<<<numBlocks, blockSize >>>(result_norm_gpu,
                                                   virdisTexture_gpu);
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::dist_to_virdis_pipelinef3(){
    if (verbose) { std::cout << "dist_to_virdis_pipeline started" << std::endl;}
    dist2virdis_pipelinef3<<<numBlocks, blockSize >>>(arr_input_gpu, dimensions,
                                                    virdisTexture_gpu);
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::reachability_to_img_pipelinef3(){
    if (verbose) { std::cout << "dist_to_virdis_pipeline started" << std::endl;}
    reachability2img_vect_pipeline<<<numBlocks, blockSize >>>(arr_input_gpu, dimensions,
                                                         virdisTexture_gpu);
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::copy_output_gpu2cpu(){
    cudaMemcpy(result.elements,
               result_gpu.elements,
               result_gpu.width * result_gpu.height * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(virdisTexture,
               virdisTexture_gpu,
               4 * rows * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    if (verbose) { std::cout << "GPU result copied to host" << std::endl;}
    cudaDeviceSynchronize();
    error_check();
}

void AutoEstimator::virdisresult_gpu2cpu(){
    cudaMemcpy(virdisTexture,
               virdisTexture_gpu,
               4 * rows * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    if (verbose) { std::cout << "virdis result copied to host" << std::endl;}
    cudaDeviceSynchronize();
    error_check();
}

void AutoEstimator::delete_all(){
    delete[] table_input.elements;
    cudaFree(table_input_gpu.elements);
    cudaFree(result_gpu.elements);
    cudaFree(result_norm_gpu.elements);
    cudaFree(virdisTexture_gpu);
    delete[] result.elements;
    delete[] result_norm.elements;
    delete[] virdisTexture;
    if (verbose) { std::cout << "All pointers deleted" << std::endl;}
    error_check();
}