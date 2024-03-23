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
#include <tuple>

int main() {

    LegDimensions (*LegToUse)(float body_angle);
    // LegToUse = get_moonbot_leg;
    LegToUse = get_M2_leg;

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
            legArray.elements[leg] = LegToUse(leg * angle);
        }

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

        std::cout << (size_t)body_pos_arr.length * (size_t)target_map.length *
                         (size_t)legArray.length
                  << std::endl;
        Array<int> out_count;
        Array<float3> out_body;
        auto start = std::chrono::high_resolution_clock::now();

        std::tie(out_body, out_count) =
            robot_full_struct(body_pos_arr, target_map, legArray);
        delete[] body_pos_arr.elements;
        body_pos_arr.length = out_body.length;
        body_pos_arr.elements = out_body.elements;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Target*Map*legs = "
                  << (size_t)body_pos_arr.length * (size_t)target_map.length *
                         (size_t)legArray.length
                  << std::endl;
        std::cout << "Cuda robot reachability took " << duration.count()
                  << " milliseconds to finish." << std::endl;

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
        saveArrayToFile(out_count.elements, out_count.length, filename);
        delete[] target_map.elements;
        delete[] body_pos_arr.elements;
        delete[] out_count.elements;
        delete[] legArray.elements;
    }
    {
        LegDimensions dim = LegToUse(0);
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
        LegDimensions dim = LegToUse(0);
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
