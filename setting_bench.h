#include <cstddef>
constexpr size_t MinSample = 100;
constexpr size_t MaxSample = 10'000'000;
constexpr int SubSamples_GPU = 100;
constexpr int SubSamples_CPU = 10;
constexpr int SubSamples_RBDL = 3;
constexpr float Spacing = 2;
constexpr float MinPix = 0.04; // bench 0.04
constexpr float MinPixRBDL = 0.4; // bench 0.4
constexpr float MaxPix = 50;

constexpr float XMin = -100;
constexpr float XMax = 601;
constexpr float YMin = 0;
constexpr float YMax = 0;
constexpr float ZMin = -350;
constexpr float ZMax = 51;
