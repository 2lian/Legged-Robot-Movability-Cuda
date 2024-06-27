#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"

float apply_RBDL(Array<float3> input, LegDimensions leg, Array<bool> output);
