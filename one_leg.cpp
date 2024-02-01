#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
#include "math_util.h"
#include "one_leg.cu.h"
#include "static_variables.h"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <ostream>

TEST_CASE("single leg reachability", "[reachability]") {
    LegDimensions dim = get_SCARE_leg(0.0f);
    Array<float3> arr{};
    Array<bool> out{};

    /*     arr.length = 10; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 10") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 100; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 100") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 1000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 1000") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 10000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 10000") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 100000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 100000") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 1000000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 1000000") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 10000000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new bool[out.length]; */
    /*     BENCHMARK("Reachability computation 10_000_000") { */
    /*         apply_kernel(arr, dim, reachability_kernel, out); */
    /*         return; */
    /*     }; */

    /*     delete[] arr.elements; */
    /*     delete[] out.elements; */

    arr.length = 100;
    arr.elements = new float3[arr.length];
    for (int i = 0; i < arr.length; i++) {
        arr.elements[i].x = 0.0f;
        arr.elements[i].y = 0.0f;
        arr.elements[i].z = 0.0f;
    }
    /* memset(arr.elements, 0, arr.length * sizeof(float3)); */
    arr.elements[0].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist + 1;

    arr.elements[1].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist - 1;

    arr.elements[2].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist + 1;

    arr.elements[3].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist - 1;

    out.length = arr.length;
    out.elements = new bool[out.length];

    apply_kernel(arr, dim, reachability_kernel, out);

    SECTION("manually placed points") {
        CHECK(arr.elements[4].x == 0);
        CHECK(arr.elements[4].y == 0);
        CHECK(arr.elements[4].z == 0);
        CHECK(out.elements[0] == true);
        CHECK(out.elements[1] == false);
        CHECK(out.elements[2] == false);
        CHECK(out.elements[3] == true);
    }

    delete[] arr.elements;
    delete[] out.elements;

    int samples_per_joint = 30;
    float angle_margin = 0.01;
    arr.length = samples_per_joint * samples_per_joint * samples_per_joint;
    arr.elements = new float3[arr.length];
    for (int i = 0; i < arr.length; i++) {
        arr.elements[i].x = 0.0f;
        arr.elements[i].y = 0.0f;
        arr.elements[i].z = 0.0f;
    }

    int counter = 0;
    for (float a1r = 0; a1r <= 1;
         a1r += 1.0f / (float)(samples_per_joint - 1)) {
        float coxa = (dim.min_angle_coxa + angle_margin) +
                     a1r * (-(dim.min_angle_coxa + angle_margin) +
                            (dim.max_angle_coxa - angle_margin));

        for (float a2r = 0; a2r <= 1;
             a2r += 1.0f / (float)(samples_per_joint - 1)) {
            float femur = (dim.min_angle_femur + angle_margin) +
                          a2r * (-(dim.min_angle_femur + angle_margin) +
                                 (dim.max_angle_femur - angle_margin));

            for (float a3r = 0; a3r <= 1;
                 a3r += 1.0f / (float)(samples_per_joint - 1)) {
                float tibia = (dim.min_angle_tibia + angle_margin) +
                              a3r * (-(dim.min_angle_tibia + angle_margin) +
                                     (dim.max_angle_tibia - angle_margin));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = femur;
                arr.elements[counter].z = tibia;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }
        }
    }

    Array<float3> intermediate;
    intermediate.length = arr.length;
    intermediate.elements = new float3[intermediate.length];
    apply_kernel(arr, dim, forward_kine_kernel, intermediate);

    out.length = arr.length;
    out.elements = new bool[out.length];
    apply_kernel(intermediate, dim, reachability_kernel, out);

    SECTION("Forward kinematics correspondence") {
        for (int i = 0; i < arr.length; i++) {
            CHECK(out.elements[i] == true);
            if (out.elements[i] != true) {
                std::cout << "angles : " << arr.elements[i].x << " | "
                          << arr.elements[i].y << " | " << arr.elements[i].z
                          << " | " << std::endl;
            }
        }
    }

    delete[] arr.elements;
    delete[] intermediate.elements;
    delete[] out.elements;
}

TEST_CASE("single leg distance", "[distance]") {
    LegDimensions dim = get_SCARE_leg(0.0f);
    Array<float3> arr{};
    Array<float3> out{};

    /*     arr.length = 10; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 10") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 100; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 100") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 1000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 1000") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 10000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 10000") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 100000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 100000") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 1000000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 1000000") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     arr.length = 10000000; */
    /*     delete[] arr.elements; */
    /*     arr.elements = new float3[arr.length]; */
    /*     memset(arr.elements, 0, arr.length * sizeof(float3)); */
    /*     out.length = arr.length; */
    /*     delete[] out.elements; */
    /*     out.elements = new float3[out.length]; */
    /*     BENCHMARK("Distance computation 10_000_000") { */
    /*         apply_kernel(arr, dim, dist_kernel, out); */
    /*         return; */
    /*     }; */

    /*     delete[] out.elements; */
    /*     delete[] arr.elements; */

    arr.length = 100;
    arr.elements = new float3[arr.length];
    for (int i = 0; i < arr.length; i++) {
        arr.elements[i].x = 0.0f;
        arr.elements[i].y = 0.0f;
        arr.elements[i].z = 0.0f;
    }

    float overshoot = 10.0f;

    arr.elements[0].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist + 1.0f;

    arr.elements[1].x =
        dim.body + dim.coxa_length + dim.min_tibia_to_gripper_dist - overshoot;

    arr.elements[2].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist + overshoot;

    arr.elements[3].x =
        dim.body + dim.coxa_length + dim.max_tibia_to_gripper_dist - 1;

    out.length = arr.length;
    out.elements = new float3[out.length];

    // Ensure memory allocation is successful
    if (arr.elements == nullptr || out.elements == nullptr) {
        std::cerr << "Memory allocation failed." << std::endl;
    }

    apply_kernel(arr, dim, dist_kernel, out);

    float interval = 0.001f;
    SECTION("manually placed points") {
        CHECK_THAT(out.elements[0].x,
                   Catch::Matchers::WithinRel((float)0.0f, interval));
        CHECK_THAT(out.elements[0].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out.elements[0].z,
                   Catch::Matchers::WithinRel(0.0f, interval));

        CHECK_THAT(out.elements[1].x,
                   Catch::Matchers::WithinRel(-overshoot, interval));
        CHECK_THAT(out.elements[1].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out.elements[1].z,
                   Catch::Matchers::WithinRel(0.0f, interval));

        CHECK_THAT(out.elements[2].x,
                   Catch::Matchers::WithinRel(overshoot, interval));
        CHECK_THAT(out.elements[2].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out.elements[2].z,
                   Catch::Matchers::WithinRel(0.0f, interval));

        CHECK_THAT(out.elements[3].x,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out.elements[3].y,
                   Catch::Matchers::WithinRel(0.0f, interval));
        CHECK_THAT(out.elements[3].z,
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
