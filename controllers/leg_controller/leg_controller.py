"""
    @file: leg_controller.py
    @author: HuNing-He
    @date: 2022-01-24
    @email: huning-he@qq.com
    @copyright(c) 2022 HuNing-He
    @brief: Anyone can use the code for secondary development, free of charge without any restrictions,
            but it is forbidden to sell the open source code without any modification.
            Hope you can keep the author information.If there exist any problems, contact me by email.
    @description: contains four main control logic for single leg: inverse-kinematic, inverse-dynamic, vmc and joint pd control
"""
import numpy as np
from controller import Robot
from kinematic import Kinematic
from FootStepPlanner import StepPlanner
from four_linked_leg import FourLinkedLeg
from dynamic import LegDynamic
import time as time_consuming
import struct

print_control_bandwidth = False
controller_switch = 2  # 0: Inverse Kinematic control; 1: VMC; 2: Inverse Dynamic control; other : joint pd control

init_motor_pos = 0.2618
knee_bias = 0.58276

leg_link = [0, 0.21, 0.2008]  # unit: m
kine = Kinematic(leg_link, is_elbow=True)

four_link = [15, 209.23, 23, 210]  # unit: mm
leg = FourLinkedLeg(four_link, knee_bias, init_motor_pos)

leg_dynamic = LegDynamic(leg)

time_swing = 0.5
time_stance = 0.5
step_planner = StepPlanner(time_swing, time_stance)

robot = Robot()
timestep = int(robot.getBasicTimeStep())

hip_motor = robot.getDevice('motor1')
knee_motor = robot.getDevice('motor2')

hip_pos = robot.getDevice('sensor1')
knee_motor_pos = robot.getDevice('sensor2')
pos_2 = robot.getDevice("pos2")
knee_pos = robot.getDevice('knee_pos_sensor')

hip_pos.enable(timestep)
knee_motor_pos.enable(timestep)
knee_pos.enable(timestep)
pos_2.enable(timestep)

emitter = robot.createEmitter("emitter")
emitter.setChannel(100)

time = 0.0
abs_time = 0.0

pre_qd = np.array([0, 0, 0])
pre_q = np.array([0, 0, 0])


def joint_pd_control():
    k_p = np.array([[0, 0, 0],
                    [0, 80, 0],
                    [0, 0, 80]], dtype=float)
    k_d = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=float)
    global pre_q, pre_qd

    theta = np.array([0, hip_pos.getValue(), knee_motor_pos.getValue()])
    theta_dot = (theta - pre_q) / 0.001 * timestep * 0.8 + 0.2 * pre_qd
    pre_qd = theta_dot
    pre_q = theta
    theta_des = np.array([0, 1.4, 0.0])
    theta_dot_des = np.array([0, 0, 0])

    torque = k_p @ (theta_des - theta) + k_d @ (theta_dot_des - theta_dot)
    for i in range(2):
        if torque[i + 1] >= 14:
            torque[i + 1] = 14
        elif torque[i + 1] <= -14:
            torque[i + 1] = -14
        else:
            torque[i + 1] = torque[i + 1]

    hip_motor.setTorque(torque[1])
    knee_motor.setTorque(torque[2])


def inverse_kinematic_control(t):
    ft, dft, ddft = step_planner.foot_traj(t)
    des_theta = kine.inverse(ft)
    q_d_knee = leg.knee_to_motor_angle(des_theta[2])
    q_d_hip = des_theta[1]
    hip_motor.setPosition(q_d_hip)
    knee_motor.setPosition(q_d_knee)


pre_foot_pos = np.array([0, 0, 0], dtype=float)
pre_foot_vel = np.array([0, 0, 0], dtype=float)


def single_leg_vmc_control(t):
    k_p = np.array([[500, 0, 0],
                    [0, 0, 0],
                    [0, 0, 500]], dtype=float)
    k_d = np.array([[60, 0, 0],
                    [0, 0, 0],
                    [0, 0, 60]], dtype=float)
    global pre_foot_pos, pre_foot_vel
    start = time_consuming.time()

    des_foot_pos, des_foot_vel, des_foot_acc = step_planner.foot_traj(t)
    theta = np.array([0, hip_pos.getValue(), leg.motor_to_knee_angle(knee_motor_pos.getValue())])
    foot_pos = np.array(kine.forward(theta))
    # test foot position error
    """
    print("foot pos calculated by forward kinematic:")
    print(foot_pos)
    """
    foot_vel = 0.8 * (foot_pos - pre_foot_pos) / (0.001 * timestep) + 0.2 * pre_foot_vel
    pre_foot_vel = foot_vel
    pre_foot_pos = foot_pos
    f = k_p @ (np.array(des_foot_pos) - foot_pos) + k_d @ (np.array(des_foot_vel) - foot_vel)

    torque = np.array(kine.jacobi(theta)).T @ f
    torque[2] *= leg.reduction_ratio(knee_motor_pos.getValue())

    for i in range(2):
        if torque[i + 1] >= 14:
            torque[i + 1] = 14
        elif torque[i + 1] <= -14:
            torque[i + 1] = -14
        else:
            torque[i + 1] = torque[i + 1]

    hip_motor.setTorque(torque[1])
    knee_motor.setTorque(torque[2])
    end = time_consuming.time()
    if print_control_bandwidth:
        print('VMC running period: %s Seconds' % (end - start))

    # After testing, it is found that the modeling error of the four-bar linkage in this code is very small
    """
    leg.update_theta(knee_motor_pos.getValue())
    print("theta2 in four-bar linkage:\n")
    print(leg.theta[1])

    print("actual theta2:\n")
    print(pos_2.getValue() - 0.19021639)
    
    print("error in theta2:\n")
    print(leg.theta[1] - pos_2.getValue() + 0.19021639)

    print("theta4 in four-bar linkage:\n")
    print(theta[2] + 3.1415926)
    
    act_knee = knee_pos.getValue()
    print("actual theta4:\n")
    print(-act_knee + 0.377660042)
    
    print("error in theta4:\n")
    print(theta[2] + 3.1415926 + act_knee - 0.377660042)
    """


pre_theta_pos = np.array([0, 0], dtype=float)
pre_theta_vel = np.array([0, 0], dtype=float)


def single_leg_inv_dynamics_control(t):
    torque = np.zeros(2)
    joint_space = False

    global pre_theta_pos, pre_theta_vel

    start = time_consuming.time()
    ft, dft, ddft = step_planner.foot_traj(t)
    theta = np.array([hip_pos.getValue(), knee_motor_pos.getValue()])
    theta_dot = 0.9 * (theta - pre_theta_pos) / (0.001 * timestep) + 0.1 * pre_theta_vel
    pre_theta_vel = theta_dot
    pre_theta_pos = theta
    leg_dynamic.update_dynamics(theta, theta_dot)

    q = np.array([0, theta[0], leg.motor_to_knee_angle(theta[1])])

    if joint_space:
        kp = np.array([[120, 0],
                       [0, 120]])
        kd = np.array([[30, 0],
                       [0, 30]])

        theta_des = np.array([kine.inverse(ft)[1], leg.knee_to_motor_angle(kine.inverse(ft)[2])])
        theta_dot_des = np.linalg.pinv(np.array(kine.jacobi(q))) @ np.array(dft)
        theta_dot_des[2] /= leg.reduction_ratio(theta_des[1])
        theta_dot_des = np.array([theta_dot_des[1], theta_dot_des[2]])
        theta_ddot = kp @ (theta_des - theta) + kd @ (theta_dot_des - theta_dot)
        torque = leg_dynamic.H @ theta_ddot + leg_dynamic.C
    else:
        kp = np.array([[300, 0, 0],
                       [0, 0, 0],
                       [0, 0, 300]])
        kd = np.array([[60, 0, 0],
                       [0, 0, 0],
                       [0, 0, 60]])

        qdot = np.array([0, theta_dot[0], theta_dot[1] * leg.reduction_ratio(theta[1])])

        act_foot = np.array(kine.forward(q))
        act_foot_vel = np.array(kine.jacobi(q)) @ qdot
        foot_ddot = kp @ (ft - act_foot) + kd @ (dft - act_foot_vel)
        torque = leg_dynamic.H @ np.linalg.pinv(leg_dynamic.J) @ (foot_ddot - leg_dynamic.JdotQdot) + leg_dynamic.C
    # with open('theta_dot.txt', 'a') as f:
    #     s = str(theta_dot[0]) + '\t' + str(theta_dot[1]) + '\n'
    #     f.writelines(s)

    for i in range(2):
        if torque[i] >= 14:
            torque[i] = 14
        elif torque[i] <= -14:
            torque[i] = -14
        else:
            torque[i] = torque[i]

    hip_motor.setTorque(torque[0])
    knee_motor.setTorque(torque[1])

    end = time_consuming.time()
    if print_control_bandwidth:
        print('Inv Dynamic Control running period: %s Seconds' % (end - start))


while robot.step(timestep) != -1:
    if controller_switch == 0:
        inverse_kinematic_control(time)
    elif controller_switch == 1:
        single_leg_vmc_control(time)
    elif controller_switch == 2:
        single_leg_inv_dynamics_control(time)
    else:
        joint_pd_control()

    abs_time += 0.001 * timestep
    time += 0.001 * timestep
    time %= time_swing + time_stance

    ft, dft, ddft = step_planner.foot_traj(time)
    message = struct.pack("3d", ft[0], ft[1] + 0.052, ft[2])
    emitter.send(message)
