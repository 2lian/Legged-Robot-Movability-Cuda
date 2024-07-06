#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cross_compiled.cuh"
#include "math_util.h"
#include "one_leg.cu.h"
#include "setting_bench.h"
// #include "several_leg.cu.h"
#include "settings.h"
#include "several_leg_octree.cu.h"
#include "static_variables.h"
// #include "vector_types.h"
#include "RBDL_benchmark.h"
#include <chrono>
#include <iostream>
#include <iterator>
#include <ostream>
// #include <tuple>
#include <fstream>
#include <vector>

std::vector<float> arange(float start, float end, float step) {
    std::vector<float> result;
    for (float value = start; value <= end; value += step) {
        result.push_back(value);
    }
    return result;
}

// Function to generate a 3D grid
Array<float3> generate3DGrid(const std::vector<float>& x_values,
                             const std::vector<float>& y_values,
                             const std::vector<float>& z_values) {
    Array<float3> out;
    // std::cout << x_values.size() << "|" << y_values.size() << "|" << z_values.size()
    // << "|" << std::endl;
    out.length = x_values.size() * y_values.size() * z_values.size();
    out.elements = new float3[out.length];
    size_t i = 0;

    for (float x_val : x_values) {
        for (float y_val : y_values) {
            for (float z_val : z_values) {
                out.elements[i] = {x_val, y_val, z_val};
                i++;
            }
        }
    }

    return out;
}

int main() {

    LegDimensions (*LegToUse)(float body_angle);
    if constexpr (RobotNumb == 0)
        LegToUse = get_moonbot_leg;
    else
        LegToUse = get_M2_leg;
    LegDimensions dim = LegToUse(0);

    for (uint computeIndex = 0; computeIndex < 4; computeIndex++) {
        bool reach;
        uint compute_mode;
        int subsample;
        const char* file;
        if (computeIndex == 0) {
            compute_mode = GPUMode;
            reach = true;
            file = "./bdata/pc/rgpu.csv";
        } else if (computeIndex == 1) {
            compute_mode = CPUMode;
            reach = true;
            file = "./bdata/pc/rcpu.csv";
        } else if (computeIndex == 2) {
            compute_mode = GPUMode;
            reach = false;
            file = "./bdata/pc/dgpu.csv";
        } else if (computeIndex == 3) {
            compute_mode = CPUMode;
            reach = false;
            file = "./bdata/pc/dcpu.csv";
        } else if (computeIndex == 4) {
            compute_mode = RBDLMode;
            reach = true;
            file = "./bdata/pc/rbdl.csv";
        } else {
            std::cerr << "Unknown compute index." << std::endl;
            return 1;
        }

        if (compute_mode == GPUMode)
            subsample = SubSamples_GPU;
        else if (compute_mode == CPUMode)
            subsample = SubSamples_CPU;

        std::ofstream csvFile(file);
        if (!csvFile.is_open()) {
            std::cerr << "Failed to open file." << std::endl;
            return 1;
        }

        // for (int P = 1; P <= Samples; P=P*2) {

        for (int sub = 0; sub < subsample; sub++) {
            for (float x = MinPix; x <= MaxPix; x = x * Spacing) {
                // float x =
                // MaxPix * ((Samples - P) / (float)Samples) + MinPix *
                // (P/(float)Samples);
                auto Pix = x;

                float x_start = XMin, x_end = XMax, x_step = Pix;
                float y_start = YMin, y_end = YMax, y_step = Pix;
                float z_start = XMin, z_end = ZMax, z_step = Pix;

                std::vector<float> x_values = arange(x_start, x_end, x_step);
                std::vector<float> y_values = arange(y_start, y_end, y_step);
                std::vector<float> z_values = arange(z_start, z_end, z_step);

                Array<float3> target_map = generate3DGrid(x_values, y_values, z_values);

                float duration;
                if (compute_mode == GPUMode and reach) {
                    Array<bool> out;
                    out.length = target_map.length;
                    out.elements = new bool[out.length];

                    duration =
                        apply_kernel(target_map, dim, reachability_global_kernel, out);
                    delete[] out.elements;
                } else if (compute_mode == CPUMode and reach) {
                    Array<bool> out;
                    out.length = target_map.length;
                    out.elements = new bool[out.length];

                    duration = apply_reach_cpu(target_map, dim, out);
                    delete[] out.elements;
                } else if (compute_mode == GPUMode and not reach) {
                    Array<float3> out;
                    out.length = target_map.length;
                    out.elements = new float3[out.length];

                    duration = apply_kernel(target_map, dim, distance_global_kernel, out);
                    delete[] out.elements;
                } else if (compute_mode == CPUMode and not reach) {
                    Array<float3> out;
                    out.length = target_map.length;
                    out.elements = new float3[out.length];

                    duration = apply_dist_cpu(target_map, dim, out);
                    delete[] out.elements;
                } else if (compute_mode == RBDLMode and reach) {
                    Array<bool> out;
                    out.length = target_map.length;
                    out.elements = new bool[out.length];

                    duration = apply_RBDL(target_map, dim, out);
                    delete[] out.elements;
                } else
                    std::cout << "Compute Mode Error" << std::endl;
                // std::cout << "Cuda reachability took " << duration
                // << " milliseconds to finish." << std::endl;
                double ns_per_point =
                    ((double)duration / (double)target_map.length) * 1'000'000.0;
                std::cout << "That's " << ns_per_point
                          << " ns per point (total: " << target_map.length << ")"
                          << std::endl;
                int intValue = target_map.length;
                double floatValue = ns_per_point;
                csvFile << intValue << ";" << floatValue << std::endl;

                delete[] target_map.elements;
                x_values.clear();
                y_values.clear();
                z_values.clear();
            }
        }
        csvFile.close();
    }
}
