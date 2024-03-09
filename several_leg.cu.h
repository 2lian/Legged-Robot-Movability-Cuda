#pragma once
#include "HeaderCUDA.h"


Array<int> robot_full_reachable(Array<float3> body_map,
                                Array<float3> target_map,
                                Array<LegDimensions> legs);
Array<int> robot_full_cccl(Array<float3> body_map,
                                Array<float3> target_map,
                                Array<LegDimensions> legs);
