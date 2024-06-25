#pragma once
#include "HeaderCUDA.h"

template <typename T_in, typename param, typename T_out>
float apply_kernel(const Array<T_in> input, const param dim,
                   void (*kernel)(const Array<T_in>, const param, Array<T_out> const),
                   Array<T_out> const output);

template <typename T_in, typename param, typename T_out>
float apply_recurs(const Array<T_in> input, const param dim, Array<T_out> const output);

__host__ double apply_reach_cpu(const Array<float3> input, const LegDimensions dim,
                                Array<bool> const output);
__host__ double apply_dist_cpu(const Array<float3> input, const LegDimensions dim,
                           Array<float3> const output);
