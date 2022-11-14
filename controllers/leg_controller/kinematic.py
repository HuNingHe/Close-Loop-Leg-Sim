"""
    @file: kinematic.py
    @author: HuNing-He
    @date: 2022-01-21
    @email: huning-he@qq.com
    @copyright(c) 2022 HuNing-He
    @brief: Anyone can use the code for secondary development, free of charge without any restrictions,
            but it is forbidden to sell the open source code without any modification.
            Hope you can keep the author information.If there exist any problems, contact me by email.
    @description: This file contains the kinematics of single leg whose coordinate system is the same as the Mini Cheetah
"""

from math import sin, cos, sqrt, pi, atan2, asin, acos
from typing import Sequence
import numpy as np
import sympy


def symbol_inverse_kinematic_print():
    print("Begin to calculate symbol inverse kinematic")
    l1, l2, l3, theta1, theta2, theta3 = sympy.symbols('l1 l2 l3 theta1 theta2 theta3')
    print("The forward kinematic as follow:")
    x = -l3 * sympy.sin(theta2 + theta3) - l2 * sympy.sin(theta2)
    y = l3 * sympy.cos(theta2 + theta3) * sympy.sin(theta1) + l2 * sympy.cos(theta2) * sympy.sin(theta1) + l1 * sympy.sin(theta1)
    z = -l3 * sympy.cos(theta2 + theta3) * sympy.cos(theta1) - l2 * sympy.cos(theta2) * sympy.cos(theta1) - l1 * sympy.cos(theta1)
    print("x = ", x)
    print("y = ", y)
    print("z = ", z)

    print("It's easy to find that theta1 = -atan2(y, z)")
    tmp1 = (y * sympy.sin(theta1) - z * sympy.cos(theta1) - l1)
    print("Observe that:tmp1 = (y * sin(theta1) - z * cos(theta1) - l1) = \n", tmp1.simplify())
    tmp2 = x ** 2 + tmp1 ** 2
    print("Firstly, calculate tmp2 = x ** 2 + tmp1 ** 2 = \n", tmp2.simplify())

    print("Now we can easily get theta3 = acos((tmp2 - l2 ** 2 - l3 ** 2)/(2*l2*l3))")
    print("Function acos is in range [0, pi]")
    print("So, if you are using elbow-config leg then theta3 = -acos((tmp2 - l2 ** 2 - l3 ** 2)/(2*l2*l3))")

    tmp3 = tmp1 * sympy.sin(theta2) + x * sympy.cos(theta2)
    print("Observe that tmp1*sin(theta2)+x*cos(theta2) = \n", tmp3.simplify())
    print("Utilizing auxiliary angle formula, theta2 can be easily solved")
    print("Result as below:")
    print("theta2 = asin(-sin(theta3) * l3 / sqrt(tmp2)) - atan2(x, tmp1)")


class Kinematic:
    def __init__(self, link_length: Sequence, is_elbow: bool):
        # thigh and leg length should bigger than 0
        assert link_length[0] >= 0
        assert link_length[1] > 0
        assert link_length[2] > 0
        self._LinkLength = link_length  # Length of Hip Thigh Leg
        self._isElbow = is_elbow  # leg type

    def forward(self, theta: Sequence) -> Sequence:
        s1 = sin(theta[0])
        c1 = cos(theta[0])
        c2 = cos(theta[1])
        s2 = sin(theta[1])
        c23 = cos(theta[1] + theta[2])
        s23 = sin(theta[1] + theta[2])
        foot_position = [0, 0, 0]

        foot_position[0] = -self._LinkLength[2] * s23 - self._LinkLength[1] * s2
        foot_position[1] = self._LinkLength[2] * c23 * s1 + self._LinkLength[1] * c2 * s1 + self._LinkLength[0] * s1
        foot_position[2] = -self._LinkLength[2] * c23 * c1 - self._LinkLength[1] * c2 * c1 - self._LinkLength[0] * c1
        return foot_position

    def jacobi(self, theta: Sequence):
        s1 = sin(theta[0])
        c1 = cos(theta[0])
        c2 = cos(theta[1])
        s2 = sin(theta[1])
        c23 = cos(theta[1] + theta[2])
        s23 = sin(theta[1] + theta[2])

        J = []
        for _ in range(3):
            J.append([])

        J[0].append(0)
        J[0].append(-self._LinkLength[2] * c23 - self._LinkLength[1] * c2)
        J[0].append(-self._LinkLength[2] * c23)

        J[1].append(self._LinkLength[2] * c23 * c1 + self._LinkLength[1] * c2 * c1 + self._LinkLength[0] * c1)
        J[1].append(-self._LinkLength[2] * s23 * s1 - self._LinkLength[1] * s2 * s1)
        J[1].append(-self._LinkLength[2] * s23 * s1)

        J[2].append(self._LinkLength[2] * c23 * s1 + self._LinkLength[1] * c2 * s1 + self._LinkLength[0] * s1)
        J[2].append(self._LinkLength[2] * s23 * c1 + self._LinkLength[1] * s2 * c1)
        J[2].append(self._LinkLength[2] * s23 * c1)
        return J

    def inverse(self, p: Sequence):
        theta1 = atan2(p[1], -p[2])
        tmp1 = p[1] * sin(theta1) - p[2] * cos(theta1) - self._LinkLength[0]
        tmp2 = p[0] ** 2 + tmp1 ** 2
        # The range of function acos is [0, pi]. Therefore, theta3 should be negative for elbow-configured leg
        if not self._isElbow:
            theta3 = acos((tmp2 - self._LinkLength[1] ** 2 - self._LinkLength[2] ** 2) / (2 * self._LinkLength[1] * self._LinkLength[2]))
        else:
            theta3 = -acos((tmp2 - self._LinkLength[1] ** 2 - self._LinkLength[2] ** 2) / (2 * self._LinkLength[1] * self._LinkLength[2]))

        theta2 = asin(-sin(theta3) * self._LinkLength[2] / sqrt(tmp2)) - atan2(p[0], tmp1)
        theta = [theta1, theta2, theta3]
        return theta


if __name__ == '__main__':
    symbol_inverse_kinematic_print()
    link_length = [0, 0.21, 0.2008]
    kin = Kinematic(link_length, is_elbow=True)

    print(np.array(kin.inverse([0, 0.1, -0.24])))
    print(np.array(kin.forward([0, pi / 2, 0])))
