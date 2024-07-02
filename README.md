# A GPU-Optimised library for quick assessment of 3DoF legged robot positionability

This library written in C++ and Cuda provides Inverse Kinematic kernel for a 3 DoF (YPP) leg optimised for GPU operation. It does not provide an IK solution but 1. if a solution exists 2. the distance to the leg reachabily volume.

<img src="https://github.com/hubble14567/Legged-Robot-Movability-Cuda/assets/70491689/91415fc3-bc50-4719-8d3a-37c029412203" width="400">

# This is a Work In Progress

A proper Python interface through PyBind11 and cleaning of the code is in progress. Before full publication I will at least provide:
- Target reachability acessement of every point in an array of 3D points.
- Target distance acessement of every point in an array of 3D points.
- Robot positionability (from a body position can the robot stand by grabbing targets with each legs) given a map as a pointcloud.
- Detailed installation explanation because compiling for CUDA is often tricky

<img src="https://github.com/hubble14567/Legged-Robot-Movability-Cuda/assets/70491689/da48f334-7c07-4208-a30b-9e8058e9d1fb" width="400">
<img src="https://github.com/hubble14567/Legged-Robot-Movability-Cuda/assets/70491689/161e7b7e-5a8e-4e5c-93ce-f452cc611ee2" width="400">
