#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
#include "one_leg.cu.h"
#include "static_variables.h"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <ostream>

TEST_CASE("single leg reachability") {
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
    SECTION("manually placed points"
            /* , "[!mayfail]" */
    ) {
        delete[] arr.elements;
        delete[] out.elements;

        arr.length = 100;
        arr.elements = new float3[arr.length];
        /* for (int i = 0; i < arr.length; i++) { */
        /*     arr.elements[i].x = 0.0f; */
        /*     arr.elements[i].y = 0.0f; */
        /*     arr.elements[i].z = 0.0f; */
        /* } */
        memset(arr.elements, 0, arr.length * sizeof(float3));

        arr.elements[0].x =
            dim.body + dim.coxa_length + dim.min_femur_to_gripper_dist + 1;

        arr.elements[1].x =
            dim.body + dim.coxa_length + dim.min_femur_to_gripper_dist - 1;

        arr.elements[2].x =
            dim.body + dim.coxa_length + dim.max_femur_to_gripper_dist + 1;

        arr.elements[3].x =
            dim.body + dim.coxa_length + dim.max_femur_to_gripper_dist - 1;

        out.length = arr.length;
        out.elements = new bool[out.length];

        apply_kernel(arr, dim, reachability_kernel, out);

        REQUIRE(arr.elements[4].x == 0);
        REQUIRE(arr.elements[4].y == 0);
        REQUIRE(arr.elements[4].z == 0);
        REQUIRE(out.elements[0] == true);
        REQUIRE(out.elements[1] == false);
        REQUIRE(out.elements[2] == false);
        REQUIRE(out.elements[3] == true);
    }

    SECTION("Forward kinematics TRUE"
            /* , "[!mayfail]" */
    ) {

        delete[] arr.elements;
        delete[] out.elements;

        int samples_per_joint = 100;
        float angle_margin = 0.0001;
        arr.length = samples_per_joint * samples_per_joint * samples_per_joint;
        arr.elements = new float3[arr.length];

        int counter = 0;
        for (int a1 = 0; a1 < samples_per_joint; a1++) {
            float a1r = (float)a1 / (float)(samples_per_joint - 1);
            float coxa = (dim.min_angle_coxa + angle_margin) +
                         a1r * (-(dim.min_angle_coxa + angle_margin) +
                                (dim.max_angle_coxa - angle_margin));

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);
                float femur = (dim.min_angle_femur + angle_margin) +
                              a2r * (-(dim.min_angle_femur + angle_margin) +
                                     (dim.max_angle_femur - angle_margin));

                for (int a3 = 0; a3 < samples_per_joint; a3++) {
                    float a3r = (float)a3 / (float)(samples_per_joint - 1);
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

        for (int i = 0; i < arr.length; i++) {
            REQUIRE(out.elements[i] == true);
            if (out.elements[i] != true) {
                std::cout << "angles : " << arr.elements[i].x << " | "
                          << arr.elements[i].y << " | " << arr.elements[i].z
                          << " | " << std::endl;
            }
        }
        delete[] intermediate.elements;
    }

    SECTION("Forward kinematics FALSE, tibia zero far"
            /* , "[!mayfail]" */
    ) {
        float tibia_elongation = 0.001;
        dim.tibia_length += tibia_elongation;

        delete[] arr.elements;
        delete[] out.elements;

        int samples_per_joint = 101;
        float angle_margin = 0.0;
        arr.length = samples_per_joint * samples_per_joint;
        arr.elements = new float3[arr.length];

        int counter = 0;
        for (int a1 = 0; a1 < samples_per_joint; a1++) {
            float a1r = (float)a1 / (float)(samples_per_joint - 1);
            float coxa = (dim.min_angle_coxa + angle_margin) +
                         a1r * (-(dim.min_angle_coxa + angle_margin) +
                                (dim.max_angle_coxa - angle_margin));

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);
                float femur = (dim.min_angle_femur + angle_margin) +
                              a2r * (-(dim.min_angle_femur + angle_margin) +
                                     (dim.max_angle_femur - angle_margin));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = femur;
                arr.elements[counter].z = 0.0f;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }
        }

        Array<float3> intermediate;
        intermediate.length = arr.length;
        intermediate.elements = new float3[intermediate.length];
        apply_kernel(arr, dim, forward_kine_kernel, intermediate);

        dim.tibia_length -= tibia_elongation;

        out.length = arr.length;
        out.elements = new bool[out.length];
        apply_kernel(intermediate, dim, reachability_kernel, out);

        for (int i = 0; i < arr.length; i++) {
            REQUIRE(out.elements[i] == false);
            if (out.elements[i] != false) {
                std::cout << "angles : " << arr.elements[i].x << " | "
                          << arr.elements[i].y << " | " << arr.elements[i].z
                          << " | " << std::endl;
            }
        }
        delete[] intermediate.elements;
    }

    SECTION("Forward kinematics FALSE, femur saturated tibia elongatred"
            /* , "[!mayfail]" */
    ) {

        float tibia_elongation = 0.01;
        dim.tibia_length += tibia_elongation;

        delete[] arr.elements;
        delete[] out.elements;

        int samples_per_joint = 101;
        float angle_margin = 0.0;
        arr.length = samples_per_joint * samples_per_joint * 2;
        arr.elements = new float3[arr.length];

        int counter = 0;
        for (int a1 = 0; a1 < samples_per_joint; a1++) {
            float a1r = (float)a1 / (float)(samples_per_joint - 1);
            float coxa = (dim.min_angle_coxa + angle_margin) +
                         a1r * (-(dim.min_angle_coxa + angle_margin) +
                                (dim.max_angle_coxa - angle_margin));

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);

                float tibia = (dim.min_angle_tibia + angle_margin) +
                              a2r * (-(dim.min_angle_tibia + angle_margin) +
                                     (0.0f - angle_margin));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = dim.min_angle_femur;
                arr.elements[counter].z = tibia;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);
                float tibia = (0.0f + angle_margin) +
                              a2r * (-(0.0f + angle_margin) +
                                     (dim.max_angle_tibia - angle_margin));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = dim.max_angle_femur;
                arr.elements[counter].z = tibia;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }
        }

        Array<float3> intermediate;
        intermediate.length = arr.length;
        intermediate.elements = new float3[intermediate.length];
        apply_kernel(arr, dim, forward_kine_kernel, intermediate);

        dim.tibia_length -= tibia_elongation; // f you you made me lose 2 days

        out.length = arr.length;
        out.elements = new bool[out.length];
        apply_kernel(intermediate, dim, reachability_kernel, out);

        for (int i = 0; i < arr.length; i++) {
            REQUIRE(out.elements[i] == false);
            if (out.elements[i] != false) {
                std::cout << "angles : " << arr.elements[i].x << " | "
                          << arr.elements[i].y << " | " << arr.elements[i].z
                          << " | " << std::endl;
                std::cout << "coord : " << intermediate.elements[i].x << " | "
                          << intermediate.elements[i].y << " | "
                          << intermediate.elements[i].z << " | " << std::endl;
            }
        }
        delete[] intermediate.elements;
    }

    SECTION("Forward kinematics FALSE, too close tibia saturated"
            /* , "[!mayfail]" */
    ) {

        delete[] arr.elements;
        delete[] out.elements;

        int samples_per_joint = 101;
        float angle_overshoot = 0.01f;
        arr.length = samples_per_joint * samples_per_joint;
        arr.elements = new float3[arr.length];

        int counter = 0;
        for (int a1 = 0; a1 < samples_per_joint; a1++) {
            float a1r = (float)a1 / (float)(samples_per_joint - 1);
            float coxa = (dim.min_angle_coxa) +
                         a1r * (-(dim.min_angle_coxa) + (dim.max_angle_coxa));

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);

                float femur =
                    (dim.min_angle_femur) +
                    a2r * (-(dim.min_angle_femur) + (dim.max_angle_femur));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = femur;
                arr.elements[counter].z = dim.min_angle_tibia - angle_overshoot;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }
        }

        Array<float3> intermediate;
        intermediate.length = arr.length;
        intermediate.elements = new float3[intermediate.length];
        apply_kernel(arr, dim, forward_kine_kernel, intermediate);

        out.length = arr.length;
        out.elements = new bool[out.length];
        apply_kernel(intermediate, dim, reachability_kernel, out);

        for (int i = 0; i < arr.length; i++) {
            REQUIRE(out.elements[i] == false);
            if (out.elements[i] != false) {
                std::cout << "angles : " << arr.elements[i].x << " | "
                          << arr.elements[i].y << " | " << arr.elements[i].z
                          << " | " << std::endl;
                std::cout << "coord : " << intermediate.elements[i].x << " | "
                          << intermediate.elements[i].y << " | "
                          << intermediate.elements[i].z << " | " << std::endl;
            }
        }
        delete[] intermediate.elements;
    }

    delete[] arr.elements;
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

    SECTION("manually placed points"
            /* , "[!mayfail]" */
    ) {
        delete[] arr.elements;
        delete[] out.elements;

        arr.length = 100;
        arr.elements = new float3[arr.length];
        for (int i = 0; i < arr.length; i++) {
            arr.elements[i].x = 0.0f;
            arr.elements[i].y = 0.0f;
            arr.elements[i].z = 0.0f;
        }
        float overshoot = 10.0f;
        arr.elements[0].x =
            dim.body + dim.coxa_length + dim.min_femur_to_gripper_dist + 1.0f;
        arr.elements[1].x = dim.body + dim.coxa_length +
                            dim.min_femur_to_gripper_dist - overshoot;
        arr.elements[2].x = dim.body + dim.coxa_length +
                            dim.max_femur_to_gripper_dist + overshoot;
        arr.elements[3].x =
            dim.body + dim.coxa_length + dim.max_femur_to_gripper_dist - 1;
        out.length = arr.length;
        out.elements = new float3[out.length];

        // Ensure memory allocation is successful
        if (arr.elements == nullptr || out.elements == nullptr) {
            std::cerr << "Memory allocation failed." << std::endl;
        }

        apply_kernel(arr, dim, dist_kernel, out);
        float interval = 0.001f;
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

    SECTION("Forward kinematics distance, too far tibia elongated and zeroed",
            "[~!mayfail]") {

        delete[] arr.elements;
        delete[] out.elements;

        float tibia_elongation = 1;
        dim.tibia_length += tibia_elongation;

        int samples_per_joint = 101;
        float angle_overshoot = 0;
        arr.length = samples_per_joint * samples_per_joint;
        arr.elements = new float3[arr.length];
        int counter = 0;
        for (int a1 = 0; a1 < samples_per_joint; a1++) {
            float a1r = (float)a1 / (float)(samples_per_joint - 1);
            float coxa = (dim.min_angle_coxa) +
                         a1r * (-(dim.min_angle_coxa) + (dim.max_angle_coxa));

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);

                float femur =
                    (dim.min_angle_femur) +
                    a2r * (-(dim.min_angle_femur) + (dim.max_angle_femur));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = femur;
                arr.elements[counter].z = 0;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }
        }

        Array<float3> intermediate;
        intermediate.length = arr.length;
        intermediate.elements = new float3[intermediate.length];
        apply_kernel(arr, dim, forward_kine_kernel, intermediate);

        dim.tibia_length -= tibia_elongation;

        out.length = arr.length;
        out.elements = new float3[out.length];
        apply_kernel(intermediate, dim, dist_kernel, out);
        float interval = 0.1;
        for (int i = 0; i < arr.length; i++) {
            REQUIRE_THAT(out.elements[i].x * out.elements[i].x +
                             out.elements[i].y * out.elements[i].y +
                             out.elements[i].z * out.elements[i].z,
                         Catch::Matchers::WithinRel(
                             tibia_elongation * tibia_elongation, interval));

            /* if (out.elements[i] != false) { */
            /*     std::cout << "angles : " << arr.elements[i].x << " | " */
            /*               << arr.elements[i].y << " | " << arr.elements[i].z
             */
            /*               << " | " << std::endl; */
            /*     std::cout << "coord : " << intermediate.elements[i].x << " |
             * " */
            /*               << intermediate.elements[i].y << " | " */
            /*               << intermediate.elements[i].z << " | " <<
             * std::endl; */
            /* } */
        }
        delete[] intermediate.elements;
    }

    SECTION("Forward kinematics distance, too far tibia elongated and zeroed",
            "[!mayfail]") {

        delete[] arr.elements;
        delete[] out.elements;

        float tibia_elongation = 1;
        dim.tibia_length += tibia_elongation;

        int samples_per_joint = 11;
        float angle_overshoot = 0;
        arr.length = samples_per_joint * 1;
        arr.elements = new float3[arr.length];
        int counter = 0;
        for (int a1 = 0; a1 < 1; a1++) {
            float a1r = (float)a1 / (float)(samples_per_joint - 1);
            a1r = 0.5;
            float coxa = (dim.min_angle_coxa) +
                         a1r * (-(dim.min_angle_coxa) + (dim.max_angle_coxa));

            for (int a2 = 0; a2 < samples_per_joint; a2++) {
                float a2r = (float)a2 / (float)(samples_per_joint - 1);

                float tibia = (0) + a2r * (-(0) + (dim.max_angle_tibia));

                arr.elements[counter].x = coxa;
                arr.elements[counter].y = dim.max_angle_femur;
                arr.elements[counter].z = tibia;

                if (counter > arr.length) {
                    std::cout << "ERROR Array overflowing" << std::endl;
                }
                counter++;
            }
        }

        Array<float3> intermediate;
        intermediate.length = arr.length;
        intermediate.elements = new float3[intermediate.length];
        apply_kernel(arr, dim, forward_kine_kernel, intermediate);

        dim.tibia_length -= tibia_elongation;

        out.length = arr.length;
        out.elements = new float3[out.length];
        apply_kernel(intermediate, dim, dist_kernel, out);
        float interval = 0.5;
        for (int i = 0; i < arr.length; i++) {
            CHECK_THAT(out.elements[i].x * out.elements[i].x +
                             out.elements[i].y * out.elements[i].y +
                             out.elements[i].z * out.elements[i].z,
                         Catch::Matchers::WithinRel(
                             tibia_elongation * tibia_elongation, interval));

            if (true != false) {
                std::cout << "angles : " << arr.elements[i].x << " | "
                          << arr.elements[i].y << " | " << arr.elements[i].z

                          << " | " << std::endl;
                std::cout << "coord : " << intermediate.elements[i].x
                          << " | " 
                          << intermediate.elements[i].y << " | "
                          << intermediate.elements[i].z << " | " << std::endl;
            }
        }
        delete[] intermediate.elements;
    }
    delete[] arr.elements;
    delete[] out.elements;
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
        dim.body + dim.coxa_length + dim.min_femur_to_gripper_dist + 1;

    arr.elements[1].x =
        dim.body + dim.coxa_length + dim.min_femur_to_gripper_dist - 1;

    arr.elements[2].x =
        dim.body + dim.coxa_length + dim.max_femur_to_gripper_dist + 1;

    arr.elements[3].x =
        dim.body + dim.coxa_length + dim.max_femur_to_gripper_dist - 1;

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
