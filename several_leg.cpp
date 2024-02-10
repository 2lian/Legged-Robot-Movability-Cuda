#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
#include "math_util.h"
#include "several_leg.cu.h"
#include "static_variables.h"
#include "vector_types.h"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <ostream>

int main() {

    const char* filename = "numpy_input_tx.bin";
    Array<float> inputxx;
    inputxx = readArrayFromFile<float>(filename);
    filename = "numpy_input_ty.bin";
    Array<float> inputxy;
    inputxy = readArrayFromFile<float>(filename);
    filename = "numpy_input_tz.bin";
    Array<float> inputxz;
    inputxz = readArrayFromFile<float>(filename);

    Array<float3> target_map = threeArrays2float3Arr(inputxx, inputxy, inputxz);
    delete[] inputxx.elements;
    delete[] inputxy.elements;
    delete[] inputxz.elements;

    Array<LegDimensions> legArray;
    legArray.length = 6;
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
    out = robot_full_reachable(body_pos_arr, target_map, legArray);
    for (int i = 0; i < body_pos_arr.length; i++) {
        if (i < target_map.length) {
            std::cout << target_map.elements[i].x << "  |  "
                      << target_map.elements[i].y << "  |  "
                      << target_map.elements[i].z << std::endl;
        }
        std::cout << body_pos_arr.elements[i].x << "  |  "
                  << body_pos_arr.elements[i].y << "  |  "
                  << body_pos_arr.elements[i].z << std::endl;
        std::cout << out.elements[i] << std::endl;
        std::cout << std::endl;
    }
    float x_arr[body_pos_arr.length];
    for (int i = 0; i < body_pos_arr.length; i++) {
        x_arr[i] = body_pos_arr.elements[i].x;
    }
    float y_arr[body_pos_arr.length];
    for (int i = 0; i < body_pos_arr.length; i++) {
        y_arr[i] = body_pos_arr.elements[i].y;
    }

    filename = "cpp_array_xx.bin";
    saveArrayToFile(x_arr, body_pos_arr.length, filename);
    filename = "cpp_array_xy.bin";
    saveArrayToFile(y_arr, body_pos_arr.length, filename);
    filename = "cpp_array_y.bin";
    saveArrayToFile(out.elements, out.length, filename);
}
