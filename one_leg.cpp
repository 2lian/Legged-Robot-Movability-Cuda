#include "one_leg.h"
#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_util.h"
#include "math_util.h"
#include "static_variables.h"
#include <iostream>
/* #include "cuda_runtime_api.h" */
/* #include <driver_types.h> */

int main(int argc, char* argv[]) {
    Arrayf3 arr{};
    arr.length = 10;
    arr.elements = new float3[arr.length];

    Arrayf3 out{};
    out.length = arr.length;
    out.elements = new float3[arr.length];

    LegDimensions dim = get_SCARE_leg(0);
    apply_kernel(arr, dim, dist_kernel, out);
    std::cout << out.elements[0].x << out.elements[0].y << out.elements[0].z;
    return 0;
}
