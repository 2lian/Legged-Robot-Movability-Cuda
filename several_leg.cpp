#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
#include "math_util.h"
#include "one_leg.cu.h"
#include "several_leg.cu.h"
#include "static_variables.h"
#include "vector_types.h"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <ostream>

int main() {

    {
        const char* filename = "numpy_input_tx.bin";
        Array<float> inputxx;
        inputxx = readArrayFromFile<float>(filename);
        filename = "numpy_input_ty.bin";
        Array<float> inputxy;
        inputxy = readArrayFromFile<float>(filename);
        filename = "numpy_input_tz.bin";
        Array<float> inputxz;
        inputxz = readArrayFromFile<float>(filename);

        Array<float3> target_map =
            threeArrays2float3Arr(inputxx, inputxy, inputxz);
        delete[] inputxx.elements;
        delete[] inputxy.elements;
        delete[] inputxz.elements;

        Array<LegDimensions> legArray;
        legArray.length = 4;
        legArray.elements = new LegDimensions[legArray.length];
        float angle = pI * 2 / legArray.length;
        for (int leg = 0; leg < legArray.length; leg++) {
            legArray.elements[leg] = get_SCARE_leg(leg * angle);
        }

        /* float z = 0; */
        /* float y_min = -500; */
        /* float y_max = 1000; */
        /* float x_min = 0; */
        /* float x_max = 2000; */
        /* float spacing = 5; */
        /* int xsamples = (x_max - x_min) / spacing + 1; */
        /* int ysamples = (y_max - y_min) / spacing + 1; */

        /* Array<float3> target_map; */
        /* target_map.length = xsamples * ysamples; */
        /* target_map.elements = new float3[target_map.length]; */

        /* long count = 0; */
        /* for (float x = x_min; x <= x_max; x += spacing) { */
        /*     for (float y = y_min; y <= y_max; y += spacing) { */
        /*         target_map.elements[count].x = x; */
        /*         target_map.elements[count].y = y; */
        /*         target_map.elements[count].z = z; */
        /*         count++; */
        /*     } */
        /* } */
        /* long target_count = count; */

        filename = "numpy_input_bx.bin";
        Array<float> body_xx;
        body_xx = readArrayFromFile<float>(filename);
        filename = "numpy_input_by.bin";
        Array<float> body_xy;
        body_xy = readArrayFromFile<float>(filename);
        filename = "numpy_input_bz.bin";
        Array<float> body_xz;
        body_xz = readArrayFromFile<float>(filename);

        Array<float3> body_pos_arr =
            threeArrays2float3Arr(body_xx, body_xy, body_xz);
        delete[] body_xx.elements;
        delete[] body_xy.elements;
        delete[] body_xz.elements;

        std::cout << (long)body_pos_arr.length * (long)target_map.length *
                         (long)legArray.length
                  << std::endl;
        Array<int> out;
        auto start = std::chrono::high_resolution_clock::now();

        out = robot_full_reachable(body_pos_arr, target_map, legArray);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Cuda robot reachability took " << duration.count()
                  << " milliseconds to finish." << std::endl;

        /* for (int i = 0; i < body_pos_arr.length; i++) { */
        /*     if (i < target_map.length) { */
        /*         std::cout << target_map.elements[i].x << "  |  " */
        /*                   << target_map.elements[i].y << "  |  " */
        /*                   << target_map.elements[i].z << std::endl; */
        /*     } */
        /*     std::cout << body_pos_arr.elements[i].x << "  |  " */
        /*               << body_pos_arr.elements[i].y << "  |  " */
        /*               << body_pos_arr.elements[i].z << std::endl; */
        /*     std::cout << out.elements[i] << std::endl; */
        /*     std::cout << std::endl; */
        /* } */
        {
            float* x_arr = new float[body_pos_arr.length];
            for (int i = 0; i < body_pos_arr.length; i++) {
                x_arr[i] = body_pos_arr.elements[i].x;
            }
            filename = "cpp_array_xx.bin";
            saveArrayToFile(x_arr, body_pos_arr.length, filename);
            delete[] x_arr;
        }
        {
            float* y_arr = new float[body_pos_arr.length];
            for (int i = 0; i < body_pos_arr.length; i++) {
                y_arr[i] = body_pos_arr.elements[i].y;
            }
            filename = "cpp_array_xy.bin";
            saveArrayToFile(y_arr, body_pos_arr.length, filename);
            delete[] y_arr;
        }
        {
            float* z_arr = new float[body_pos_arr.length];
            for (int i = 0; i < body_pos_arr.length; i++) {
                z_arr[i] = body_pos_arr.elements[i].z;
            }
            filename = "cpp_array_xz.bin";
            saveArrayToFile(z_arr, body_pos_arr.length, filename);
            delete[] z_arr;
        }

        filename = "cpp_array_y.bin";
        saveArrayToFile(out.elements, out.length, filename);
        delete[] target_map.elements;
        delete[] body_pos_arr.elements;
        delete[] out.elements;
        delete[] legArray.elements;
    }
    {
        LegDimensions dim = get_SCARE_leg(0);
        const char* filename = "dist_input_tx.bin";
        Array<float> inputxx = readArrayFromFile<float>(filename);
        filename = "dist_input_ty.bin";
        Array<float> inputxy = readArrayFromFile<float>(filename);
        filename = "dist_input_tz.bin";
        Array<float> inputxz = readArrayFromFile<float>(filename);

        Array<float3> target_map =
            threeArrays2float3Arr(inputxx, inputxy, inputxz);
        delete[] inputxx.elements;
        delete[] inputxy.elements;
        delete[] inputxz.elements;

        Array<float3> out2;
        out2.length = target_map.length;
        out2.elements = new float3[out2.length];

        auto start = std::chrono::high_resolution_clock::now();

        apply_kernel(target_map, dim, dist_kernel, out2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Cuda distance took " << duration.count()
                  << " milliseconds to finish." << std::endl;

        delete[] target_map.elements;

        {
            float* x_arr2 = new float[out2.length];
            for (int i = 0; i < out2.length; i++) {
                x_arr2[i] = out2.elements[i].x;
            }
            filename = "out_dist_xx.bin";
            saveArrayToFile(x_arr2, out2.length, filename);
            delete[] x_arr2;
        }
        {
            float* y_arr2 = new float[out2.length];
            for (long i = 0; i < out2.length; i++) {
                y_arr2[i] = out2.elements[i].y;
            }
            filename = "out_dist_xy.bin";
            saveArrayToFile(y_arr2, out2.length, filename);
            delete[] y_arr2;
        }
        {
            float* z_arr2 = new float[out2.length];
            for (long i = 0; i < out2.length; i++) {
                z_arr2[i] = out2.elements[i].z;
            }
            filename = "out_dist_xz.bin";
            saveArrayToFile(z_arr2, out2.length, filename);
            delete[] z_arr2;
        }
    }
    {
        LegDimensions dim = get_SCARE_leg(0);
        const char* filename = "dist_input_tx.bin";
        Array<float> inputxx = readArrayFromFile<float>(filename);
        filename = "dist_input_ty.bin";
        Array<float> inputxy = readArrayFromFile<float>(filename);
        filename = "dist_input_tz.bin";
        Array<float> inputxz = readArrayFromFile<float>(filename);

        Array<float3> target_map =
            threeArrays2float3Arr(inputxx, inputxy, inputxz);
        delete[] inputxx.elements;
        delete[] inputxy.elements;
        delete[] inputxz.elements;

        Array<bool> out2;
        out2.length = target_map.length;
        out2.elements = new bool[out2.length];

        auto start = std::chrono::high_resolution_clock::now();

        apply_kernel(target_map, dim, reachability_abs_tib_kernel, out2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Cuda reachability took " << duration.count()
                  << " milliseconds to finish." << std::endl;

        delete[] target_map.elements;

        filename = "out_reachability.bin";
        saveArrayToFile(out2.elements, out2.length, filename);
    }
}
