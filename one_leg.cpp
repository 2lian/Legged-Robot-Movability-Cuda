#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
#include "math_util.h"
#include "one_leg.cu.h"
#include "static_variables.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>

TEST_CASE("single leg reachability", "[reachability]") {
    LegDimensions dim = get_SCARE_leg(0);

    Array<float3> arr{};
    arr.length = 10;
    arr.elements = new float3[arr.length];
    memset(arr.elements, 0, arr.length * sizeof(float3));

    arr.elements[0].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist + 1;

    arr.elements[1].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist - 1;

    arr.elements[2].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist + 1;

    arr.elements[3].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist - 1;

    Array<bool> out1{};
    out1.length = arr.length;
    out1.elements = new bool[out1.length];

    apply_kernel(arr, dim, reachability_kernel, out1);

    SECTION("manually placed points") {
        REQUIRE(out1.elements[0] == true);
        REQUIRE(out1.elements[1] == false);
        REQUIRE(out1.elements[2] == false);
        REQUIRE(out1.elements[3] == true);
    }
}

TEST_CASE("single leg distance", "[distance]") {
    LegDimensions dim = get_SCARE_leg(0);

    Array<float3> arr{};
    arr.length = 10;
    arr.elements = new float3[arr.length];
    memset(arr.elements, 0, arr.length * sizeof(float3));

    float overshoot = 10.0f;
    arr.elements[0].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist + 1;

    arr.elements[1].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist - overshoot;

    arr.elements[2].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist + overshoot;

    arr.elements[3].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist - 1;

    Array<float3> out1{};
    out1.length = arr.length;
    out1.elements = new float3[out1.length];

    apply_kernel(arr, dim, dist_kernel, out1);

    float interval = 0.001;
    SECTION("manually placed points") {
        CHECK_THAT(out1.elements[0].x,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out1.elements[0].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out1.elements[0].z,
                   Catch::Matchers::WithinRel(0.0f, interval));

        CHECK_THAT(out1.elements[1].x,
                   Catch::Matchers::WithinRel(-overshoot, interval));
        CHECK_THAT(out1.elements[1].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out1.elements[1].z,
                   Catch::Matchers::WithinRel(0.0f, interval));

        CHECK_THAT(out1.elements[2].x,
                   Catch::Matchers::WithinRel(overshoot, interval));
        CHECK_THAT(out1.elements[2].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out1.elements[2].z,
                   Catch::Matchers::WithinRel(0.0f, interval));

        CHECK_THAT(out1.elements[3].x,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out1.elements[3].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out1.elements[3].z,
                   Catch::Matchers::WithinRel(0.0f, interval));
    }
}

/**
 * @brief
 *
 * @param argc
 * @param argv
 * @return
 */
int mainnot(int argc, char* argv[]) {
    LegDimensions dim = get_SCARE_leg(0);

    Array<float3> arr{};
    arr.length = 10;
    arr.elements = new float3[arr.length];

    arr.elements[0].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist + 1;

    arr.elements[1].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist - 1;

    arr.elements[2].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist + 1;

    arr.elements[3].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist - 1;

    Array<float3> out1{};
    out1.length = arr.length;
    out1.elements = new float3[out1.length];

    apply_kernel(arr, dim, dist_kernel, out1);
    std::cout << out1.elements[0].x << std::endl;
    std::cout << out1.elements[1].x << std::endl;
    std::cout << out1.elements[2].x << std::endl;
    std::cout << out1.elements[3].x << std::endl;

    Array<bool> out2{};
    out2.length = arr.length;
    out2.elements = new bool[out2.length];

    apply_kernel(arr, dim, reachability_kernel, out2);
    std::cout << out2.elements[0] << std::endl;
    std::cout << out2.elements[1] << std::endl;
    std::cout << out2.elements[2] << std::endl;
    std::cout << out2.elements[3] << std::endl;
    return 0;
}
