"""
    @file: four_linked_leg.py
    @author: HuNing-He
    @date: 2022-01-23
    @email: huning-he@qq.com
    @copyright(c) 2022 HuNing-He
    @brief: Anyone can use the code for secondary development, free of charge without any restrictions,
            but it is forbidden to sell the open source code without any modification.
            Hope you can keep the author information.If there exist any problems, contact me by email.
    @description: This file contains motion analysis of four linked leg, all angular units being radians
"""

from math import cos, sin, sqrt, asin, atan2, pi
import numpy as np
# curve fitting, The reduction ratio formula increases the complexity of calculation, so we use the polynomial curve to replace it
from numpy import polyfit
import matplotlib.pyplot as plt
from typing import Sequence
import sympy


class FourLinkedLeg:
    def __init__(self, _link_length: Sequence, _knee_bias: float = 0.0, _motor_correction: float = 0.0):
        """
        Args
            :param _link_length: 4 x 1 array, represent the four link length, refer to paper: 四足机器人腿部结构运动学分析与仿真, 李栓成等
            :param _knee_bias: a fixed bias between link and actual leg
            :param _motor_correction: init angle of motor, defined by exported 3D model, webots regard this position as zero position,
                   but it's not the actual zero position, so that we need to make some corrections(default to zero)
        """
        self.link_length = _link_length
        self.knee_bias = _knee_bias
        self.motor_correction = _motor_correction
        self.theta = np.zeros(4, dtype=float)

    def update_theta(self, theta: float):
        """
        Args
            :param theta: actual motor position
        """
        theta += self.motor_correction
        a = 2 * self.link_length[0] * self.link_length[2] * sin(theta)
        b = 2 * self.link_length[2] * (self.link_length[0] * cos(theta) - self.link_length[3])
        c = self.link_length[1] ** 2 - self.link_length[0] ** 2 - self.link_length[2] ** 2 - self.link_length[3] ** 2 + 2 * \
            self.link_length[0] * self.link_length[3] * cos(theta)

        self.theta[0] = theta
        self.theta[3] = asin(-c / sqrt(a ** 2 + b ** 2)) - atan2(b, a)
        if self.link_length[0] * sin(self.theta[0]) <= self.link_length[2] * sin(self.theta[3]):
            self.theta[1] = asin(
                (self.link_length[2] * sin(self.theta[3]) - self.link_length[0] * sin(theta)) / self.link_length[1]) - theta
        else:
            self.theta[1] = -asin(
                (self.link_length[2] * sin(self.theta[3]) - self.link_length[0] * sin(theta)) / self.link_length[1]) - theta
        self.theta[2] = self.theta[3] - pi - theta - self.theta[1]

    def motor_to_knee_angle(self, theta):
        """
        Args
            :param theta: actual motor position
            :return: desire knee position, should be negative in knee-configured leg,
                     which depends on the joint coordinate when building kinematics
        """
        theta += self.motor_correction
        a = 2 * self.link_length[0] * self.link_length[2] * sin(theta)
        b = 2 * self.link_length[2] * (self.link_length[0] * cos(theta) - self.link_length[3])
        c = self.link_length[1] ** 2 - self.link_length[0] ** 2 - self.link_length[2] ** 2 - self.link_length[3] ** 2 + 2 * \
            self.link_length[0] * self.link_length[3] * cos(theta)
        return asin(-c / sqrt(a ** 2 + b ** 2)) - atan2(b, a) - self.knee_bias - pi

    def knee_to_motor_angle(self, theta: float):
        """
        Args
            :param theta: desire knee angle(negative in knee-configured leg), should first convert to the third link's angle,
                          this is easy, exchange l1 and l3 link, and according to the paper, we can derive: pi - theta3 =  motor_to_knee_angle(pi - theta1)
        """
        theta = pi + theta + self.knee_bias
        a = 2 * self.link_length[0] * self.link_length[2] * sin(theta)
        b = -2 * self.link_length[0] * (self.link_length[2] * cos(theta) + self.link_length[3])
        c = self.link_length[1] ** 2 - self.link_length[0] ** 2 - self.link_length[2] ** 2 - self.link_length[3] ** 2 - 2 * \
            self.link_length[2] * self.link_length[3] * cos(theta)
        return pi - (asin(-c / sqrt(a ** 2 + b ** 2)) - atan2(b, a)) - self.motor_correction

    def print_symbol_diff_angle(self):
        # l1, l2, l3, l4, theta1 = sympy.symbols('l1 l2 l3 l4 theta1')
        theta1 = sympy.symbols('theta1')
        l1, l2, l3, l4 = self.link_length[0], self.link_length[1], self.link_length[2], self.link_length[3]
        c = l2 ** 2 - l1 ** 2 - l3 ** 2 - l4 ** 2 + 2 * l1 * l4 * sympy.cos(theta1)
        b = 2 * l3 * (l1 * sympy.cos(theta1) - l4)
        a = 2 * l1 * l3 * sympy.sin(theta1)
        theta4_symbol = sympy.asin(-c / sympy.sqrt(a ** 2 + b ** 2)) - sympy.atan2(b, a)
        theta2_symbol = sympy.asin((l3 * sympy.sin(theta4_symbol) - l1 * sympy.sin(theta1)) / l2) - theta1
        theta3_symbol = theta4_symbol - sympy.pi - theta1 - theta2_symbol
        print("theta4 = \n", theta4_symbol.simplify())
        theta4_dot = theta4_symbol.diff(theta1).simplify()
        print("Differentiate theta4 about theta1 is : \n", theta4_dot)
        print("Second Differentiate theta4 about theta1 is : \n", theta4_dot.diff(theta1).simplify())

        print("theta2 = \n", theta2_symbol.simplify())
        theta2_dot = theta2_symbol.diff(theta1).simplify()
        print("Differentiate theta2 about theta1 is ： \n", theta2_dot)
        print("Second Differentiate theta2 about theta1 is : \n", theta2_dot.diff(theta1))

        print("With theta2 and theta4, we can easily get theta3")

    def reduction_ratio(self, theta1, order: int = 4):
        theta1 += self.motor_correction
        if order == -1:
            # generated by print_symbol_jacobi(), just copy it here
            return 15 * (sqrt((39312900 * sin(theta1) ** 2 - 9066420 * cos(theta1) + 53479971) / (43906 - 6270 * cos(theta1))) * (
                    15 - 209 * cos(theta1)) * (43906 - 6270 * cos(theta1)) ** 2 + 209 * (43906 - 6270 * cos(theta1)) ** (3 / 2) * (
                                 87477 - 6270 * cos(theta1)) * sin(theta1)) / (
                           sqrt((39312900 * sin(theta1) ** 2 - 9066420 * cos(theta1) + 53479971) / (43906 - 6270 * cos(theta1))) * (
                           43906 - 6270 * cos(theta1)) ** 3)
        elif order == 2:
            # 2-order polynomial fitting
            return -0.24889879 * theta1 ** 2 + 0.79389053 * theta1 + 0.03116565
        elif order == 3:
            # 3-order polynomial fitting
            return 0.0260247 * theta1 ** 3 - 0.36927345 * theta1 ** 2 + 0.9533875 * theta1 - 0.02398126
        else:
            # 4-order polynomial fitting (the best fitting)
            return -0.03899938 * theta1 ** 4 + 0.26654165 * theta1 ** 3 - 0.86966069 * theta1 ** 2 + 1.3528932 * theta1 - 0.12091342

    def theta2_diff(self, theta1, order: int = 3):
        theta1 += self.motor_correction
        a = 2 * self.link_length[0] * self.link_length[2] * sin(theta1)
        b = 2 * self.link_length[2] * (self.link_length[0] * cos(theta1) - self.link_length[3])
        c = self.link_length[1] ** 2 - self.link_length[0] ** 2 - self.link_length[2] ** 2 - self.link_length[3] ** 2 + 2 * \
            self.link_length[0] * self.link_length[3] * cos(theta1)

        theta4 = asin(-c / sqrt(a ** 2 + b ** 2)) - atan2(b, a)
        if order == -1:
            # generated by print_symbol_jacobi(), just copy it here
            return (-sqrt(
                (44100.0 * sin(theta1) ** 2 + 263.299400000076 * cos(theta1) + 58824.6516326551) / (197 - 28 * cos(theta1))) * sqrt(
                1 - 0.0120839177881595 * (0.652173913043478 * sin(theta1) + sin(
                    asin((9.1304347826087 * cos(theta1) - 1.56059000000001) / sqrt(197 - 28 * cos(theta1))) + atan2(
                        690 * cos(theta1) - 9660, 690 * sin(theta1)))) ** 2) * (197 - 28 * cos(theta1)) ** 3 - 0.071691440042059 * sqrt(
                (44100.0 * sin(theta1) ** 2 + 263.299400000076 * cos(theta1) + 58824.6516326551) / (197 - 28 * cos(theta1))) * (
                            197 - 28 * cos(theta1)) ** 3 * cos(theta1) + 0.00732845831541048 * (sqrt(
                (44100.0 * sin(theta1) ** 2 + 263.299400000076 * cos(theta1) + 58824.6516326551) / (197 - 28 * cos(theta1))) * (
                                                                                                        15 - 210 * cos(theta1)) * (
                                                                                                        197 - 28 * cos(
                                                                                                    theta1)) ** 2 + 7 * (
                                                                                                        197 - 28 * cos(theta1)) ** (
                                                                                                        3 / 2) * (
                                                                                                        87573.1929 - 6300.0 * cos(
                                                                                                    theta1)) * sin(theta1)) * cos(
                asin((9.1304347826087 * cos(theta1) - 1.56059000000001) / sqrt(197 - 28 * cos(theta1))) + atan2(690 * cos(theta1) - 9660,
                                                                                                                690 * sin(theta1)))) / (
                           sqrt((44100.0 * sin(theta1) ** 2 + 263.299400000076 * cos(theta1) + 58824.6516326551) / (
                                   197 - 28 * cos(theta1))) * sqrt(1 - 0.0120839177881595 * (0.652173913043478 * sin(theta1) + sin(
                       asin((9.1304347826087 * cos(theta1) - 1.56059000000001) / sqrt(197 - 28 * cos(theta1))) + atan2(
                           690 * cos(theta1) - 9660, 690 * sin(theta1)))) ** 2) * (197 - 28 * cos(theta1)) ** 3)

        elif order == 2:
            # 2-order polynomial fitting
            if self.link_length[0] * sin(theta1) <= self.link_length[2] * sin(theta4):
                return -0.00218291 * theta1 ** 2 + 0.04170418 * theta1 - 1.06273506
            else:
                return 0.00218291 * theta1 ** 2 - 0.04170418 * theta1 - 0.93726494
        else:
            # 3-order polynomial fitting (the best fitting)
            if self.link_length[0] * sin(theta1) <= self.link_length[2] * sin(theta4):
                return 0.00983792 * theta1 ** 3 - 0.04768722 * theta1 ** 2 + 0.10199759 * theta1 - 1.08358182
            else:
                return -0.00983792 * theta1 ** 3 + 0.04768722 * theta1 ** 2 - 0.10199759 * theta1 - 0.91641818

    def theta3_diff(self, theta1):
        return self.reduction_ratio(theta1) - 1 - self.theta2_diff(theta1)

    def theta2_second_diff(self, theta1, order: int = 4):
        theta1 += self.motor_correction
        a = 2 * self.link_length[0] * self.link_length[2] * sin(theta1)
        b = 2 * self.link_length[2] * (self.link_length[0] * cos(theta1) - self.link_length[3])
        c = self.link_length[1] ** 2 - self.link_length[0] ** 2 - self.link_length[2] ** 2 - self.link_length[3] ** 2 + 2 * \
            self.link_length[0] * self.link_length[3] * cos(theta1)

        theta4 = asin(-c / sqrt(a ** 2 + b ** 2)) - atan2(b, a)
        if order == 5:
            # 5-order polynomial fitting
            if self.link_length[0] * sin(theta1) <= self.link_length[2] * sin(theta4):
                return 0.00216058 * theta1 ** 5 - 0.02110023 * theta1 ** 4 + 0.07644986 * theta1 ** 3 - 0.09738406 * theta1 ** 2 - 0.00075387 * theta1 + 0.07739031
            else:
                return -0.00216058 * theta1 ** 5 + 0.02110023 * theta1 ** 4 - 0.07644986 * theta1 ** 3 + 0.09738406 * theta1 ** 2 + 0.00075387 * theta1 - 0.07739031
        else:
            # 4-order polynomial fitting (the best fitting)
            if self.link_length[0] * sin(theta1) <= self.link_length[2] * sin(theta4):
                return -0.00444435 * theta1 ** 4 + 0.02903867 * theta1 ** 3 - 0.03646225 * theta1 ** 2 - 0.03502981 * theta1 + 0.08388728
            else:
                return 0.00444435 * theta1 ** 4 - 0.02903867 * theta1 ** 3 + 0.03646225 * theta1 ** 2 + 0.03502981 * theta1 - 0.08388728

    def theta3_second_diff(self, theta1):
        return self.theta4_second_diff(theta1) - self.theta2_second_diff(theta1)

    def theta4_second_diff(self, theta1, order: int = 3):
        theta1 += self.motor_correction
        if order == -1:
            return (0.0630064286335418 * (sqrt(
                (-263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) / (28 * cos(theta1) - 197)) * (
                                               28 * cos(theta1) - 197) ** 2 * (210 * cos(theta1) - 15) + 7 * (197 - 28 * cos(theta1)) ** (
                                               3 / 2) * (6300.0 * cos(theta1) - 87573.1929) * sin(theta1)) * (
                         -1234800.0 * sin(theta1) ** 2 - 17375400.0 * cos(theta1) + 4168560.22751436) * (
                         -0.00325564802672687 * cos(theta1) + 0.272644141951355 * cos(2 * theta1) - 1) ** 2 * sin(theta1) + 5.6 * (sqrt(
                (-263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) / (28 * cos(theta1) - 197)) * (28 * cos(
                theta1) - 197) ** 2 * (210 * cos(theta1) - 15) + 7 * (197 - 28 * cos(theta1)) ** (3 / 2) * (6300.0 * cos(
                theta1) - 87573.1929) * sin(theta1)) * (
                         0.749685697679831 * sin(theta1) ** 2 + 0.00447600440788521 * cos(theta1) + 1) ** 2 * (
                         -263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) * sin(
                theta1) - 0.0630064286335418 * (15 * sqrt(
                (-263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) / (28 * cos(theta1) - 197)) * (
                                                            14 * cos(theta1) - 1) * (28 * cos(theta1) - 197) * (
                                                            -1234800.0 * sin(theta1) ** 2 - 17375400.0 * cos(
                                                        theta1) + 4168560.22751436) * sin(theta1) + 2 * (
                                                            44100.0 * sin(theta1) ** 2 + 263.299400000076 * cos(
                                                        theta1) + 58824.6516326551) * (-840 * sqrt(
                (-263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) / (28 * cos(theta1) - 197)) * (
                                                                                                   14 * cos(theta1) - 1) * (
                                                                                                   28 * cos(theta1) - 197) * sin(
                theta1) - 210 * sqrt(
                (-263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) / (28 * cos(theta1) - 197)) * (
                                                                                                   28 * cos(theta1) - 197) ** 2 * sin(
                theta1) + 7 * (197 - 28 * cos(theta1)) ** (3 / 2) * (6300.0 * cos(theta1) - 87573.1929) * cos(theta1) - 44100.0 * (
                                                                                                   197 - 28 * cos(theta1)) ** (3 / 2) * sin(
                theta1) ** 2 + sqrt(197 - 28 * cos(theta1)) * (1852200.0 * cos(theta1) - 25746518.7126) * sin(theta1) ** 2)) * (
                         28 * cos(theta1) - 197) * (-0.00325564802672687 * cos(theta1) + 0.272644141951355 * cos(2 * theta1) - 1) ** 2) / (
                        ((-263.299400000076 * cos(theta1) + 22050.0 * cos(2 * theta1) - 80874.6516326551) / (28 * cos(theta1) - 197)) ** (
                            3 / 2) * (28 * cos(theta1) - 197) ** 5 * (
                                    0.749685697679831 * sin(theta1) ** 2 + 0.00447600440788521 * cos(theta1) + 1) ** 2)

        elif order == 4:
            # 4-order polynomial fitting (the best fitting)
            return -0.01465024 * theta1 ** 4 - 0.038468 * theta1 ** 3 + 0.44298574 * theta1 ** 2 - 1.28247211 * theta1 + 1.1837779
        else:
            # 3-order polynomial fitting (the best fitting)
            return -0.12881896 * theta1 ** 3 + 0.63128425 * theta1 ** 2 - 1.43355435 * theta1 + 1.22085901


if __name__ == '__main__':
    knee_bias = 0.58276  # The angle between third link and leg is 33.39 degree
    init_motor_pos = 0.2618  # When exporting the 3D model, the motor has been turned 15 degrees
    link_length = [15, 209.23, 23, 210]
    leg = FourLinkedLeg(link_length, knee_bias, init_motor_pos)

    leg.print_symbol_diff_angle()

    leg.update_theta(0)
    knee_test = leg.motor_to_knee_angle(0)
    print("The init knee theta in kinematic frame is: ", knee_test)
    print("The theta now is: ", leg.theta)
    print("Test for inverse convert: leg.knee_to_motor_angle(leg.motor_to_knee_angle(0)) = ", leg.knee_to_motor_angle(knee_test))

    x = np.linspace(0, 2.56, 500)  # The range of joint angle is 0.2618-2.822

    theta4 = np.array([leg.motor_to_knee_angle(x[i]) for i in range(x.size)])
    theta4_dot = np.array([leg.reduction_ratio(x[i], order=-1) for i in range(x.size)])
    theta4_dot_fitted_4_order = np.array([leg.reduction_ratio(x[i]) for i in range(x.size)])
    theta4_dot_fitted_3_order = np.array([leg.reduction_ratio(x[i], 3) for i in range(x.size)])
    theta4_dot_fitted_2_order = np.array([leg.reduction_ratio(x[i], 2) for i in range(x.size)])

    theta4_second_dot = np.array([leg.theta4_second_diff(x[i], -1) for i in range(x.size)])
    theta4_second_dot_fit = np.array([leg.theta4_second_diff(x[i]) for i in range(x.size)])
    theta4_ddot_fit3_coeff = polyfit(x + init_motor_pos, theta4_second_dot, 3)
    print("coefficients of theta4_ddot's 3order-polynomial curve fitting: \n", theta4_ddot_fit3_coeff)

    theta2_dot = np.array([leg.theta2_diff(x[i], order=-1) for i in range(x.size)])
    theta2_dot_fit3 = np.array([leg.theta2_diff(x[i], order=3) for i in range(x.size)])
    theta2_fit3_coeff = polyfit(x + init_motor_pos, theta2_dot, 3)
    print("coefficients of theta2_dot's 3order-polynomial curve fitting: \n", theta2_fit3_coeff)

    theta4_fit2_coeff = polyfit(x + init_motor_pos, theta4_dot, 2)
    theta4_fit3_coeff = polyfit(x + init_motor_pos, theta4_dot, 3)
    theta4_fit4_coeff = polyfit(x + init_motor_pos, theta4_dot, 4)
    print("coefficients of theta4_dot's 2order-polynomial curve fitting: \n", theta4_fit2_coeff)
    print("coefficients of theta4_dot's 3order-polynomial curve fitting: \n", theta4_fit3_coeff)
    print("coefficients of theta4_dot's 4order-polynomial curve fitting: \n", theta4_fit4_coeff)

    # you can also export the data in reduction.txt to matlab, then using cftool to fitting the reduction
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.plot(x, theta4_second_dot, label='theta4_second_dot')
    ax1.plot(x, theta4_second_dot_fit, label='3-ord-theta4_ddot-fit', linestyle='--')

    # ax1.plot(x, theta4, label='theta4')
    # # ax1.plot(x, -0.092866667 * (x + 0.2618) ** 3 + 0.4429 * (x + 0.2618) ** 2 - 0.0297 * (x + 0.2618) - 2.7639326114464042, label='2-ord-theta3-fit')
    # ax1.plot(x, -0.007799876 * (x + 0.2618) ** 5 + 0.066635413 * (x + 0.2618) ** 4 - 0.289886897 * (x + 0.2618) ** 3 + 0.6764466 * (
    #         x + 0.2618) ** 2 - 0.12091342 * (x + 0.2618) - 2.776, label='5-ord-theta4-fit', linestyle='--')

    ax2.plot(x, theta4_dot, label='theta4_dot')
    # ax2.plot(x, theta4_dot_fitted_2_order, label='2-ord-theta4_dot-fit', linestyle='-.')
    # ax2.plot(x, theta4_dot_fitted_3_order, label='3-ord-theta4_dot-fit', linestyle=':')
    ax2.plot(x, theta4_dot_fitted_4_order, label='4-ord-theta4_dot-fit', linestyle='--')

    ax3.plot(x, theta2_dot, label='theta2_dot')
    ax3.plot(x, theta2_dot_fit3, label='3ord-theta2_dot-fit', linestyle='--')
    ax1.set_xlabel('theta1')
    ax1.set_ylabel('theta4_pos')
    ax1.set_title("Theta4_pos Plot")

    ax2.set_xlabel('theta1')
    ax2.set_ylabel('theta4_dot')
    ax2.set_title("Theta4_dot Plot")

    ax3.set_xlabel('theta1')
    ax3.set_ylabel('theta2_dot')
    ax3.set_title("Theta2_dot Plot")

    ax1.legend();
    ax2.legend();
    ax3.legend();
    plt.show()

    theta1 = sympy.symbols("theta1")
    f1 = 0.00983792 * theta1 ** 3 - 0.04768722 * theta1 ** 2 + 0.10199759 * theta1 - 1.08358182
    print(f1.diff(theta1))
    f2 = -0.03899938 * theta1 ** 4 + 0.26654165 * theta1 ** 3 - 0.86966069 * theta1 ** 2 + 1.3528932 * theta1 - 0.12091342
    print(f2.diff(theta1))
