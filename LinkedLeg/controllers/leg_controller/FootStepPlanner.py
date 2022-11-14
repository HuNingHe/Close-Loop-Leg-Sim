"""
    @file: FootStepPlanner.py
    @author: HuNing-He
    @date: 2022-01-11
    @email: huning-he@qq.com
    @copyright(c) 2022 HuNing-He
    @brief: Anyone can use the code for secondary development, free of charge without any restrictions,
            but it is forbidden to sell the open source code without any modification.
            Hope you can keep the author information.If there exist any problems, contact me by email.
    @description: contains swing trajectory planner
"""
from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from kinematic import Kinematic


class StepPlanner:
    def __init__(self, t_swing=0.5, t_stance=0.5, step_size=0.08, step_height=0.04):
        self.Tw = t_swing
        self.Ts = t_stance
        self.step_size = step_size
        self.step_height = step_height

    def foot_traj(self, time: float):
        """
        refer to https://blog.csdn.net/weixin_41045354/article/details/105219092
        Args
            :param time:
            :return: desire foot trajectory's position, velocity and acceleration
        """
        init_pos = [-self.step_size / 2, 0, -0.28]
        ft = [0.0, 0.0, 0.0]
        dft = [0.0, 0.0, 0.0]
        ddft = [0.0, 0.0, 0.0]

        if time <= self.Tw:
            ft[0] = self.step_size * (time / self.Tw - sin(2 * pi * time / self.Tw) / (2 * pi)) + init_pos[0]
            dft[0] = self.step_size * (1 - cos(2 * pi * time / self.Tw)) / self.Tw
            ddft[0] = self.step_size * sin(2 * pi * time / self.Tw) * 2 * pi / (self.Tw ** 2)
            if time <= self.Tw / 2:
                ft[2] = self.step_height * (2 * time / self.Tw - sin(4 * pi * time / self.Tw) / (2 * pi)) + init_pos[2]
                dft[2] = 2 * self.step_height * (1 - cos(4 * pi * time / self.Tw)) / self.Tw
                ddft[2] = 8 * self.step_height * sin(4 * pi * time / self.Tw) * pi / (self.Tw ** 2)
            else:
                ft[2] = self.step_height * (-2 * time / self.Tw + sin(4 * pi * time / self.Tw) / (2 * pi) + 2) + init_pos[2]
                dft[2] = 2 * self.step_height * (-1 + cos(4 * pi * time / self.Tw)) / self.Tw
                ddft[2] = -8 * self.step_height * sin(4 * pi * time / self.Tw) * pi / (self.Tw ** 2)
        else:
            ft[2] = init_pos[2]
            dft[2] = 0
            ddft[2] = 0

            ft[0] = (-self.step_size) * ((time - self.Tw) / self.Ts - sin(2 * pi * (time - self.Tw) / self.Ts) / (2 * pi)) + self.step_size + init_pos[0]
            dft[0] = -self.step_size * (1 - cos(2 * pi * (time - self.Tw) / self.Ts)) / self.Ts
            ddft[0] = -self.step_size * sin(2 * pi * (time - self.Tw) / self.Ts) * 2 * pi / (self.Tw ** 2)
        return [ft, dft, ddft]


if __name__ == "__main__":
    leg_link = [0, 0.21, 0.2008]  # unit: m
    kin = Kinematic(leg_link, is_elbow=True)
    time_swing = 0.5
    time_stance = 0.5
    step_planner = StepPlanner(time_swing, time_stance)
    t = np.linspace(0, 1, 100)
    knee_theta = np.array([kin.inverse(step_planner.foot_traj(t[i])[0])[2] for i in range(100)])
    z = np.array([step_planner.foot_traj(t[i])[0][2] for i in range(100)])

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(t, knee_theta, label='knee_theta')
    ax1.set_xlabel('time')
    ax1.set_ylabel('knee_theta')
    ax1.set_title("knee theta plot")
    ax1.legend();

    ax2.plot(t, z, label='z')
    ax2.set_xlabel('time')
    ax2.set_ylabel('z')
    ax2.set_title("z plot")
    ax2.legend();

    plt.show()
