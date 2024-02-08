#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
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
    float angle = pI * 2 / 4;
    for (int leg = 0; leg < legArray.length; leg++) {
        legArray.elements[leg] = get_SCARE_leg(angle);
    }

    float z = 0;
    float y_min = -400;
    float y_max = 400;
    float x_min = 0;
    float x_max = 1000;
    float spacing = 20;
    int xsamples = (x_max - x_min) / spacing + 1;
    int ysamples = (y_max - y_min) / spacing + 1;

    Array<float3> target_map;
    target_map.length = xsamples * ysamples;
    target_map.elements = new float3[target_map.length];

    int count = 0;
    for (float x = x_min; x < x_max; x += spacing) {
        for (float y = y_min; y < y_max; y += spacing) {
            target_map.elements[count].x = x;
            target_map.elements[count].y = y;
            target_map.elements[count].z = z;
            count++;
        }
    }

    float z_body = 250;
    float y_min_body = 0;
    float y_max_body = 0;
    float x_min_body = -500;
    float x_max_body = 1500;
    float spacing_body = 0.3;
    int xsamples_body = (x_max_body - x_min_body) / spacing_body + 1;

    Array<float3> body_pos_arr;
    body_pos_arr.length = xsamples_body;
    body_pos_arr.elements = new float3[body_pos_arr.length];
    count = 0;
    for (float x_body = x_min_body; x_body < x_max_body;
         x_body += spacing_body) {
        body_pos_arr.elements[count].x = x_body;
        body_pos_arr.elements[count].y = 0;
        body_pos_arr.elements[count].z = z_body;
        count++;
    }
    Array<int> out;
    out = robot_full_reachable(body_pos_arr, target_map, legArray);
    for (int i = 0; i < body_pos_arr.length; i++) {
        std::cout << out.elements[i] << std::endl;
    }
}
