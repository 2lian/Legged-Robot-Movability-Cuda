#pragma once
#include "HeaderCUDA.h"
#include <driver_types.h>

__global__ void dist_kernel(const Arrayf3 input, LegDimensions dimensions,
                               Arrayf3 const output);

__global__ void reachability_kernel(const Arrayf3 input, LegDimensions dimensions,
                               Arrayb const output);
