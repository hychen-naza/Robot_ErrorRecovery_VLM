#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
import numpy as np 
import argparse
# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

from waypoints import READY_POSITION, ROTATION_READY_POSITION, HIGH_POSITION, B_POSITION, READY_POSITION_OLD, TOP_BOTTOM_CABINET_START, MIDDLE_CABINET_START_TOP, MIDDLE_CABINET_START_BOTTOM, BOTTLE_PICK_2D

state_dict = {
    "reseted": False,
}

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def ee_delta_control_cartesian_step(base, base_cyclic, ee_pose):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()
    # TODO: check if feedback.base.tool_pose is the current tcp
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + ee_pose[0]         # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y + ee_pose[1]      # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z + ee_pose[2]      # (meters)
    # TODO: check if using euler angle is the best way 
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + ee_pose[3]# (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y + ee_pose[4]# (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z + ee_pose[5]# (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def get_cur_pos(base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    return (feedback.base.tool_pose_x, feedback.base.tool_pose_y, feedback.base.tool_pose_z)

def step_joint_tcp_rot_90(base, base_cyclic, rotation_degree=0):
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Rotate TCP vectical gripping"
    action.application_data = ""
    
    joint_angles = base.GetMeasuredJointAngles()
    current_joints = [joint_angle.value for joint_angle in joint_angles.joint_angles]
    current_joints[0] += rotation_degree
    actuator_count = base.GetActuatorCount()
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = current_joints[joint_id]
    
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def step_cartesian_abs_movement(base, base_cyclic, abs_ee_pose):
    
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = abs_ee_pose[0]          # (meters)
    cartesian_pose.y = abs_ee_pose[1]   # (meters)
    cartesian_pose.z = abs_ee_pose[2]    # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z  # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def reset_joint_state(base, is_rotation=False, special_type=None):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    ready_positin = READY_POSITION
    if is_rotation:
        ready_positin = ROTATION_READY_POSITION
    if special_type == "bottom" or special_type == "top":
        ready_positin = TOP_BOTTOM_CABINET_START
    elif special_type =="middle_top":
        ready_positin = MIDDLE_CABINET_START_TOP
    elif special_type =="middle_bottom":
        ready_positin = MIDDLE_CABINET_START_BOTTOM
    elif special_type == "bottle_pick":
        ready_positin = BOTTLE_PICK_2D
        
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = ready_positin[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def gpt_callee(base, base_cyclic, conn, is_rotation=False):
    while True:
        print("Waiting for command...")
        command = conn.recv()  # Receive command from the pipe
        action_buffer = np.zeros(6)
        if "##step relative##:" in command:
            if state_dict["reseted"]:
                state_dict["reseted"] = False
            action = eval(command.split("##step relative##:")[1])
            action = np.asarray(action)
            # if action is 2 dimensional, it is a x,y movement only 
            if action.shape[0] == 2:
                action_buffer[0] = action[0]
                action_buffer[1] = action[1]
            elif action.shape[0] == 3:
                action_buffer[0] = action[0]
                action_buffer[1] = action[1]
                action_buffer[2] = action[2]
            print("Executing relative action: ", action)
            success = ee_delta_control_cartesian_step(base, base_cyclic, action_buffer)
            response = f"Action attempted: {success}"
        elif "##rotate angle##" in command:
            if state_dict["reseted"]:
                state_dict["reseted"] = False
            rotate_degree = eval(command.split("##rotate angle##:")[1])
            action = float(rotate_degree)
            print("Executing rotate 90 action: ", action)
            success = step_joint_tcp_rot_90(base, base_cyclic, action)
            response = f"Action attempted: {success}"
        elif "##query position##" in command:
            response = f"Current position: {get_cur_pos(base_cyclic)}"
        elif "##reset##" in command:
            success_step = step_cartesian_abs_movement(base, base_cyclic, HIGH_POSITION)
            time.sleep(1)
            success_reset = reset_joint_state(base, is_rotation=is_rotation)
            if success_reset:
                state_dict["reseted"] = True
            response = f"Action attempted: {success_reset}"
        elif "##backB##" in command:
            print(f"in backB \n\n\n")
            success_step = step_cartesian_abs_movement(base, base_cyclic, B_POSITION)
            time.sleep(1)
            success_reset = reset_joint_state(base, is_rotation=is_rotation, special_type="bottle_pick")
            if success_reset and success_step:
                state_dict["reseted"] = True
            response = f"Action attempted: {success_reset and success_step}"
        elif "##query reset##" in command:
            response = f"reseted: {state_dict['reseted']}"
        elif "exit" in command:
            response = "Exiting..."
            conn.send(response)  # Send response before exiting
            break
        else:
            response = f"Received unknown command: {command}"
        conn.send(response)  # Send response back to the caller

def kinova_control(conn, connection_kwargs, is_rotate=False):
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    connection_namespace = argparse.Namespace(**connection_kwargs)
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(connection_namespace) as router:

        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Example core
        success = True

        gpt_callee(base, base_cyclic, conn, is_rotate)
            
        # You can also refer to the 110-Waypoints examples if you want to execute
        # a trajectory defined by a series of waypoints in joint space or in Cartesian space

        return 0 if success else 1

