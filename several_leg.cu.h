#pragma once
#include "HeaderCUDA.h"

Array<int> robot_full_reachable(Array<float3> body_map,
                                Array<float3> target_map,
                                Array<LegDimensions> legs);

std::tuple<Array<float3>, Array<int>>
robot_full_cccl(Array<float3> body_map, Array<float3> target_map,
                Array<LegDimensions> legs);
