"""
    @file: dynamic.py
    @author: HuNing-He
    @date: 2022-01-15
    @email: huning-he@qq.com
    @copyright(c) 2022 HuNing He
    @brief: Anyone can use the code for secondary development, free of charge without any restrictions,
            but it is forbidden to sell the open source code without any modification.
            Hope you can keep the author information.If there exist any problems, contact me by email.
    @description: dynamic equation for closed-loop single leg
"""

import sys
sys.path.append('/home/hun/rbdl-orb/build/python')
from rbdl import Model, Body, Joint, SpatialTransform, CompositeRigidBodyAlgorithm, NonlinearEffects, CalcPointAcceleration, CalcPointJacobian
import numpy as np
from four_linked_leg import FourLinkedLeg


class LegDynamic:
    def __init__(self, four_linked_leg: FourLinkedLeg):
        self.leg = four_linked_leg
        self.model = Model()
        self.model.gravity = np.array([0., 0., -9.81])
        rotor_inertial = np.array([[0.00000531, 0.00000088, -0.00000027],
                                   [0.00000088, 0.00000524, -0.00000055],
                                   [-0.00000027, -0.00000055, 0.00000492]], dtype=float)

        link_inertial = np.array([[0.00052681, 0, -0.00000897],
                                  [0, 0.00052772, 0],
                                  [-0.00000897, 0, 0.00000112]], dtype=float)

        leg_inertial = np.array([[0.00069448, 0., 0.00034033],
                                 [0., 0.00087332, 0.],
                                 [0.00034033, 0., 0.00018219]], dtype=float)

        thigh_inertial = np.array([[0.0047894, 0.0001697, -0.0003733],
                                   [0.0001697, 0.0040178, 0.00109],
                                   [-0.0003733, 0.00109, 0.0019091]], dtype=float)

        toe_inertial = np.eye(3) * 1e-06

        rotor = Body.fromMassComInertia(0.03020152, np.array([-0.0040243, 0.00796815, 0.0031742]), rotor_inertial)
        link = Body.fromMassComInertia(0.03515229, np.array([-0.00353189, 0., -0.10193327]), link_inertial)
        leg = Body.fromMassComInertia(0.07978664, np.array([0.03169095, 0., -0.07562750]), leg_inertial)
        thigh = Body.fromMassComInertia(1.028914, np.array([-0.003147, 0.031419, -0.02041768]), thigh_inertial)
        toe = Body.fromMassComInertia(0.02, np.zeros(3), toe_inertial)
        joint_rot_y = Joint.fromJointType("JointTypeRevoluteY")
        fixed_joint = Joint.fromJointType("JointTypeFixed")

        xtrans = SpatialTransform()
        xtrans.r = np.array([0., 0., 0.])
        p1 = self.model.AddBody(0, xtrans, joint_rot_y, thigh)
        xtrans.r = np.array([0., 0.046, 0.])
        p2 = self.model.AddBody(p1, xtrans, joint_rot_y, rotor)
        xtrans.r = np.array([0., 0.0065, -0.015])
        p3 = self.model.AddBody(p2, xtrans, joint_rot_y, link)
        xtrans.r = np.array([0., 0., -0.20923])
        p4 = self.model.AddBody(p3, xtrans, joint_rot_y, leg)
        xtrans.r = np.array([0.11038982, 0, -0.19050112])
        self.model.AddBody(p4, xtrans, fixed_joint, toe, "toe")

        self.JdotQdot = np.zeros(3)
        self.J = np.zeros((3, 2))
        self.H = np.zeros((2, 2))
        self.C = np.zeros(2)

    def update_dynamics(self, y, yd):
        # theta: [0-2] is hip theta, hip theta dot and hip theta dot dot, [3-5] is knee motor theta, it's dot and dot dot
        G = np.zeros((self.model.qdot_size, 2), dtype=float)
        G[0][0] = 1
        G[1][1] = 1
        G[2][1] = self.leg.theta2_diff(y[1])
        G[3][1] = self.leg.reduction_ratio(y[1]) - G[2][1] - 1
        g = np.zeros(self.model.q_size, dtype=float)
        g[2] = self.leg.theta2_second_diff(y[1]) * yd[1] ** 2
        g[3] = self.leg.theta3_second_diff(y[1]) * yd[1] ** 2

        q = np.zeros(self.model.q_size)
        qddot = np.zeros(self.model.qdot_size)
        qdot = G @ np.array([yd[0], yd[1]])

        self.leg.update_theta(y[1])

        q[0] = y[0]
        q[1] = self.leg.theta[0]
        q[2] = self.leg.theta[1]
        q[3] = self.leg.theta[2]

        H = np.zeros((self.model.qdot_size, self.model.qdot_size), dtype=float)
        CompositeRigidBodyAlgorithm(self.model, q, H, False)
        C = np.zeros(self.model.qdot_size)
        NonlinearEffects(self.model, q, qdot, C)
        self.H = (G.T @ H) @ G
        self.C = G.T @ (C + H @ g)

        self.JdotQdot = CalcPointAcceleration(self.model, q, qdot, qddot, self.model.GetBodyId("toe"), np.zeros(3), True)
        J = np.zeros((3, self.model.qdot_size))
        CalcPointJacobian(self.model, q, self.model.GetBodyId("toe"), np.zeros(3), J)
        self.J[:, 0] = J[:, 0]
        self.J[:, 1] = J[:, 1] + G[2][1] * J[:, 2] + G[3][1] * J[:, 3]


if __name__ == "__main__":
    knee_bias = 0.58276  # The angle between third link and leg is 33.39 degree
    init_motor_pos = 0.2618  # When exporting the 3D model, the motor has been turned 15 degrees
    link_length = [15, 209.23, 23, 210]
    four_link = FourLinkedLeg(link_length, knee_bias, init_motor_pos)

    leg = LegDynamic(four_link)
    leg.update_dynamics([0, 0], [0, 0])
