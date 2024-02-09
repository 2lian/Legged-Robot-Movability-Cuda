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
    Array<LegDimensions> legArray;
    legArray.length = 4;
    legArray.elements = new LegDimensions[legArray.length];
    float angle = pI * 2 / legArray.length;
    for (int leg = 0; leg < legArray.length; leg++) {
        legArray.elements[leg] = get_SCARE_leg(leg * angle);
    }

    float z = 0;
    float y_min = -500;
    float y_max = 1000;
    float x_min = 0;
    float x_max = 2000;
    float spacing = 5;
    int xsamples = (x_max - x_min) / spacing + 1;
    int ysamples = (y_max - y_min) / spacing + 1;

    Array<float3> target_map;
    target_map.length = xsamples * ysamples;
    target_map.elements = new float3[target_map.length];

    int count = 0;
    for (float x = x_min; x <= x_max; x += spacing) {
        for (float y = y_min; y <= y_max; y += spacing) {
            target_map.elements[count].x = x;
            target_map.elements[count].y = y;
            target_map.elements[count].z = z;
            count++;
        }
    }

    float z_body = 250;
    float y_min_body = 0;
    float y_max_body = 1000;
    float x_min_body = 750;
    float x_max_body = 2000;
    float spacing_body = 10;
    int xsamples_body = (x_max_body - x_min_body) / spacing_body + 1;
    int ysamples_body = (y_max_body - y_min_body) / spacing_body + 1;

    std::cout << xsamples_body << std::endl;
    std::cout << ysamples_body << std::endl;

    Array<float3> body_pos_arr;
    body_pos_arr.length = xsamples_body * ysamples_body;
    body_pos_arr.elements = new float3[body_pos_arr.length];
    count = 0;
    for (float x_body = x_min_body; x_body <= x_max_body;
         x_body += spacing_body) {
        for (float y_body = y_min_body; y_body <= y_max_body;
             y_body += spacing_body) {
            body_pos_arr.elements[count].x = x_body;
            body_pos_arr.elements[count].y = y_body;
            body_pos_arr.elements[count].z = z_body;
            count++;
        }
    }
    Array<int> out;
    out = robot_full_reachable(body_pos_arr, target_map, legArray);
    /* for (int i = 0; i <= body_pos_arr.length; i++) { */

    /*     std::cout << body_pos_arr.elements[i].x << "  |  " */
    /*               << body_pos_arr.elements[i].y << "  |  " */
    /*               << body_pos_arr.elements[i].z << std::endl; */
    /*     std::cout << out.elements[i] << std::endl; */
    /*     std::cout << std::endl; */
    /* } */
    float x_arr[body_pos_arr.length];
    for (int i = 0; i < body_pos_arr.length; i++) {
        x_arr[i] = body_pos_arr.elements[i].x;
    }
    float y_arr[body_pos_arr.length];
    for (int i = 0; i < body_pos_arr.length; i++) {
        y_arr[i] = body_pos_arr.elements[i].y;
    }

    const char* filename = "cpp_array_xx.bin";
    saveArrayToFile(x_arr, body_pos_arr.length, filename);
    filename = "cpp_array_xy.bin";
    saveArrayToFile(y_arr, body_pos_arr.length, filename);
    filename = "cpp_array_y.bin";
    saveArrayToFile(out.elements, out.length, filename);
}
