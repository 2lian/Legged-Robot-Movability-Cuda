#pragma once
#include "HeaderCUDA.h"

__global__ void dist_kernel(const Array<float3> input,
                            const LegDimensions dimensions,
                            Array<float3> const output);

__global__ void reachability_kernel(const Array<float3> input,
                                    const LegDimensions dimensions,
                                    Array<bool> const output);

/* __global__ void dist_kernel(const Arrayf3 input, LegDimensions dimensions, */
/*                             Arrayf3 const output); */

/* __global__ void reachability_kernel(const Arrayf3 input, */
/*                                     const LegDimensions dimensions, */
/*                                     Arrayb const output); */
