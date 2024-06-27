/*
 * RBDL - Rigid Body Dynamics Library
 * Copyright (c) 2011-2016 Martin Felis <martin@fysx.org>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

#include "RBDL_benchmark.h"
#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include <iostream>

#include <rbdl/rbdl.h>

using namespace RigidBodyDynamics;
using namespace RigidBodyDynamics::Math;

float apply_RBDL(Array<float3> input, LegDimensions leg, Array<bool> output) {
    rbdl_check_api_version(RBDL_API_VERSION);

    Model* model = NULL;

    unsigned int body_a_id, body_b_id, body_c_id, body_d_id;
    Body body_a, body_b, body_c, body_d;
    Joint joint_a, joint_b, joint_c, joint_d;

    model = new Model();

    // model->gravity = Vector3d(0., -9.81, 0.);
    constexpr int fact = 400;

    body_a = Body();
    joint_a = Joint(JointTypeRevoluteZ);

    body_a_id =
        model->AddBody(0, Xtrans(Vector3d(leg.body, 0., 0.) / fact), joint_a, body_a);

    body_b = Body();
    joint_b = Joint(JointTypeRevoluteY);

    body_b_id = model->AddBody(
        body_a_id, Xtrans(Vector3d(leg.coxa_length, 0., 0.) / fact), joint_b, body_b);

    body_c = Body();
    joint_c = Joint(JointTypeRevoluteY);

    body_c_id = model->AddBody(
        body_b_id, Xtrans(Vector3d(leg.femur_length, 0., 0.) / fact), joint_c, body_c);

    body_d = Body();
    joint_d = Joint(JointTypeFixed);

    body_d_id = model->AddBody(
        body_c_id, Xtrans(Vector3d(leg.tibia_length, 0., 0.) / fact), joint_d, body_d);

    unsigned int body_id = body_d_id;
    Vector3d target_pos = Vector3d::Zero();
    // Vector3d base_point_position = Vector3d::Zero();
    Vector3d base_point_position = Vector3d(0., 0., 0.);
    auto Cs = InverseKinematicsConstraintSet();
    VectorNd res2 = Vector3d::Zero();

    VectorNd Q = VectorNd::Zero(model->q_size);
    VectorNd QDot = VectorNd::Zero(model->qdot_size);
    VectorNd Tau = VectorNd::Zero(model->qdot_size);
    VectorNd QDDot = VectorNd::Zero(model->qdot_size);

    auto Qinit = 1.57 * Vector3d(0., 0., 1.);
    UpdateKinematics(*model, Qinit, QDot, QDDot);
    Vector3d res =
        CalcBodyToBaseCoordinates(*model, Qinit, body_id, base_point_position) * 1;
    // std::cout << res.transpose() << std::endl;

    target_pos = Vector3d(500., 000., 0.) / fact;
    Cs.AddPointConstraint(body_id, base_point_position, target_pos);
    Cs.num_steps = 100;
    // Cs.error_norm = 100;
    // Cs.delta_q_norm = 100;
    UpdateKinematics(*model, Qinit, QDot, QDDot);
    bool valid = InverseKinematics(*model, Vector3d(1, 1, 1), Cs, res2);

    // std::cout << valid << std::endl;
    // std::cout << res2.transpose() << std::endl;

    Cs.max_steps = 10;
    // Cs.num_steps = 10;
    int substep = 5;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < input.length; i++) {
        target_pos =
            Vector3d(input.elements[i].x, input.elements[i].y, input.elements[i].z) /
            fact;
        Cs.ClearConstraints();
        Cs.AddPointConstraint(body_id, base_point_position, target_pos);
        Vector3d start = Vector3d::Zero();
        bool valid = false;
        for (int s = 0; s < substep; s++) {
            valid = InverseKinematics(*model, start, Cs, res2);
            if (valid)
                break;
            start = Vector3d::Random() * 3.14 * 2;
        }
        output.elements[i] = valid;
    }
    auto end = std::chrono::high_resolution_clock::now();

    delete model;

    std::chrono::duration<double> duration = (end - start) * 1000;
    return duration.count();
}
int mainTest(int argc, char* argv[]) {
    rbdl_check_api_version(RBDL_API_VERSION);

    Model* model = NULL;

    unsigned int body_a_id, body_b_id, body_c_id, body_d_id;
    Body body_a, body_b, body_c, body_d;
    Joint joint_a, joint_b, joint_c, joint_d;

    model = new Model();

    // model->gravity = Vector3d(0., -9.81, 0.);

    body_a = Body();
    joint_a = Joint(JointTypeRevoluteZ);

    body_a_id = model->AddBody(0, Xtrans(Vector3d(1., 0., 0.)), joint_a, body_a);

    body_b = Body();
    joint_b = Joint(JointTypeRevoluteY);

    body_b_id = model->AddBody(body_a_id, Xtrans(Vector3d(1., 0., 0.)), joint_b, body_b);

    body_c = Body();
    joint_c = Joint(JointTypeRevoluteY);

    body_c_id = model->AddBody(body_b_id, Xtrans(Vector3d(1., 0., 0.)), joint_c, body_c);

    body_d = Body();
    joint_d = Joint(JointTypeFixed);

    body_d_id = model->AddBody(body_c_id, Xtrans(Vector3d(1., 0., 0.)), joint_d, body_d);

    VectorNd Q = VectorNd::Zero(model->q_size);
    VectorNd QDot = VectorNd::Zero(model->qdot_size);
    VectorNd Tau = VectorNd::Zero(model->qdot_size);
    VectorNd QDDot = VectorNd::Zero(model->qdot_size);

    // ForwardDynamics(*model, Q, QDot, Tau, QDDot);

    // std::cout << QDDot.transpose() << std::endl;

    auto Qinit = 1.57 * Vector3d(0, 1, 1);
    unsigned int body_id = body_d_id;
    Vector3d body_point = Vector3d::Zero();
    Vector3d target_pos = Vector3d::Zero();
    // Vector3d base_point_position = Vector3d::Zero();
    Vector3d base_point_position = Vector3d(0, 0, 0);
    auto Cs = InverseKinematicsConstraintSet();

    UpdateKinematics(*model, Qinit, QDot, QDDot);
    Vector3d res =
        CalcBodyToBaseCoordinates(*model, Qinit, body_id, base_point_position) * 1;
    std::cout << res.transpose() << std::endl;
    // Vector3d res =
    // CalcBaseToBodyCoordinates(*model, Qinit, body_id, base_point_position) * 1;

    target_pos = res;
    target_pos = Vector3d(3, 0, 1);
    Cs.AddPointConstraint(body_id, base_point_position, target_pos);
    // Cs.max_steps = 100;
    Cs.num_steps = 1;
    // std::vector<unsigned int> idVect = {body_id};
    VectorNd res2 = Vector3d::Zero();
    // std::vector<Math::Vector3d> body_points = {body_point};
    // InverseKinematics(*model, Q, idVect, base_point_position, target_pos, res2,
    // 0.00000000001, 0.01, (unsigned int)55);
    bool valid = InverseKinematics(*model, Vector3d::Zero(), Cs, res2);

    std::cout << valid << std::endl;
    std::cout << res2.transpose() << std::endl;

    res = CalcBaseToBodyCoordinates(*model, res2, body_id, base_point_position);
    std::cout << res.transpose() << std::endl;

    delete model;

    return 0;
}
