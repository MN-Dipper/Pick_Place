# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cracker_box")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # print('object_height',object.data.root_pos_w[:, 2])
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

# def object_is_lifted(
#     env: ManagerBasedRLEnv, 
#     minimal_height: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("cracker_box")
# ) -> torch.Tensor:
#     object: RigidObject = env.scene[object_cfg.name]
#     current_height = object.data.root_pos_w[:, 2]
    
#     # 计算相对于初始高度的提升量
#     initial_height = object.data.default_root_state[:, 2]  # 获取初始高度
#     height_gain = current_height - initial_height
    
#     # 计算目标高度增量
#     target_height_gain = minimal_height - initial_height
    
#     # 归一化奖励：从0到1线性增长
#     reward = height_gain / target_height_gain
#     reward = torch.clamp(reward, min=0.0, max=1.0)
    
#     return reward


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cracker_box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cracker_box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_close_threshold: float = 0.025,
    max_ee_object_distance: float = 0.03,  # 新增：末端执行器与物体的最大距离
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose when gripper is closed and close to object."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    
    # distance of the desired position to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    
    # 计算末端执行器与物体的距离
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(object.data.root_pos_w - ee_w, dim=1)
    
    # 获取夹爪关节位置
    gripper_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    
    # 判断夹爪是否闭合
    gripper_closed = gripper_joint_pos.mean(dim=1) < gripper_close_threshold
    
    # 判断末端执行器是否足够接近物体
    ee_close_to_object = object_ee_distance < max_ee_object_distance
    
    # 同时满足：夹爪闭合 AND 末端执行器接近物体
    valid_grasp = gripper_closed & ee_close_to_object
    # 只有在有效抓取时才给予距离奖励
    return valid_grasp.float() * (1 - torch.tanh(distance / std))




def close_gripper_when_near_object_smooth(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cracker_box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for closing the gripper when near the object with smooth transition.
    Unlike the binary threshold version, this function provides a smooth reward that
    gradually increases as the end-effector approaches the object.
    Args:
        env: The environment instance.
        threshold: The distance scale for the smooth transition (acts as standard deviation).
        open_joint_pos: The joint position value when the gripper is fully open.
        object_cfg: Configuration for the target object. Defaults to SceneEntityCfg("cracker_box").
        ee_frame_cfg: Configuration for the end-effector frame. Defaults to SceneEntityCfg("ee_frame").
        robot_cfg: Configuration for the robot asset containing gripper joints.
            Defaults to SceneEntityCfg("robot").
    Returns:
        The reward tensor of shape (num_envs,). Uses tanh kernel for smooth transition.
    """
    # Extract the used quantities
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get object and end-effector positions
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get gripper joint positions
    gripper_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    print('gripper_joint_pos',gripper_joint_pos)
    # print('gripper_joint_pos:', gripper_joint_pos)
    # Calculate distance between end-effector and object
    distance = torch.norm(object_pos_w - ee_pos_w, dim=-1, p=2)
    # Smooth proximity factor using tanh (closer = higher value)
    proximity_factor = 1 - torch.tanh(distance / threshold)
    # print('proximity_factor:', proximity_factor)
    # Reward for closing gripper
    closing_reward = torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)
    # print('closing_reward:', closing_reward)
    # Combine proximity and closing reward
    return proximity_factor * closing_reward


def close_gripper_when_near_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cracker_box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for closing the gripper when the end-effector is close to the object.
    This function encourages the robot to close its gripper when it is within a certain distance
    threshold from the target object.
    Args:
        env: The environment instance.
        threshold: The distance threshold below which the gripper should be closed.
        open_joint_pos: The joint position value when the gripper is fully open.
            Assumes zero joint position corresponds to fully closed gripper.
        object_cfg: Configuration for the target object. Defaults to SceneEntityCfg("cracker_box").
        ee_frame_cfg: Configuration for the end-effector frame. Defaults to SceneEntityCfg("ee_frame").
        robot_cfg: Configuration for the robot asset containing gripper joints.
            Defaults to SceneEntityCfg("robot").
    Returns:
        The reward tensor of shape (num_envs,). Returns positive reward when close to object
        and gripper is closing.
    Note:
        The reward is computed as the sum of (open_joint_pos - current_joint_pos) for all gripper joints,
        which means more closed fingers yield higher rewards.
    """
    # Extract the used quantities
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get object and end-effector positions
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    
    # Get gripper joint positions
    gripper_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (num_envs, num_gripper_joints)

    # Calculate distance between end-effector and object
    distance = torch.norm(object_pos_w - ee_pos_w, dim=-1, p=2)  # (num_envs,)
    
    # Check if end-effector is close enough to the object
    is_close = distance <= threshold  # (num_envs,)
    
    # Reward for closing gripper (difference between open position and current position)
    # Larger value means gripper is more closed
    closing_reward = torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)  # (num_envs,)
    
    # Only give reward when close to the object
    return is_close * closing_reward


# def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
#     r"""Reward the robot for reaching the drawer handle using inverse-square law.

#     It uses a piecewise function to reward the robot for reaching the handle.

#     .. math::

#         reward = \begin{cases}
#             2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
#             (1 / (1 + distance^2))^2 & \text{otherwise}
#         \end{cases}

#     """
#     ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
#     handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]

#     # Compute the distance of the end-effector to the handle
#     distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)

#     # Reward the robot for reaching the handle
#     reward = 1.0 / (1.0 + distance**2)
#     reward = torch.pow(reward, 2)
#     return torch.where(distance <= threshold, 2 * reward, reward)


# def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Reward for aligning the end-effector with the handle.

#     The reward is based on the alignment of the gripper with the handle. It is computed as follows:

#     .. math::

#         reward = 0.5 * (align_z^2 + align_x^2)

#     where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
#     and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
#     """
#     ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
#     handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]

#     ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
#     handle_mat = matrix_from_quat(handle_quat)

#     # get current x and y direction of the handle
#     handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
#     # get current x and z direction of the gripper
#     ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

#     # make sure gripper aligns with the handle
#     # in this case, the z direction of the gripper should be close to the -x direction of the handle
#     # and the x direction of the gripper should be close to the -y direction of the handle
#     # dot product of z and x should be large
#     align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


# def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Bonus for correct hand orientation around the handle.

#     The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
#     """
#     # Target object position: (num_envs, 3)
#     handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
#     # Fingertips position: (num_envs, n_fingertips, 3)
#     ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
#     lfinger_pos = ee_fingertips_w[..., 0, :]
#     rfinger_pos = ee_fingertips_w[..., 1, :]

#     # Check if hand is in a graspable pose
#     is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

#     # bonus if left finger is above the drawer handle and right below
#     return is_graspable


# def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
#     """Reward the robot's gripper reaching the drawer handle with the right pose.

#     This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
#     (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
#     """
#     # Target object position: (num_envs, 3)
#     handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
#     # Fingertips position: (num_envs, n_fingertips, 3)
#     ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
#     lfinger_pos = ee_fingertips_w[..., 0, :]
#     rfinger_pos = ee_fingertips_w[..., 1, :]

#     # Compute the distance of each finger from the handle
#     lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
#     rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

#     # Check if hand is in a graspable pose
#     is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

#     return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


# def grasp_handle(
#     env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward for closing the fingers when being close to the handle.

#     The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
#     The :attr:`open_joint_pos` is the joint position when the fingers are open.

#     Note:
#         It is assumed that zero joint position corresponds to the fingers being closed.
#     """
#     ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
#     handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
#     gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

#     distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
#     is_close = distance <= threshold

#     return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


# def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Bonus for opening the drawer given by the joint position of the drawer.

#     The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
#     """
#     drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
#     is_graspable = align_grasp_around_handle(env).float()

#     return (is_graspable + 1.0) * drawer_pos


# def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Multi-stage bonus for opening the drawer.

#     Depending on the drawer's position, the reward is given in three stages: easy, medium, and hard.
#     This helps the agent to learn to open the drawer in a controlled manner.
#     """
#     drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
#     is_graspable = align_grasp_around_handle(env).float()

#     open_easy = (drawer_pos > 0.01) * 0.5
#     open_medium = (drawer_pos > 0.2) * is_graspable
#     open_hard = (drawer_pos > 0.3) * is_graspable

#     return open_easy + open_medium + open_hard