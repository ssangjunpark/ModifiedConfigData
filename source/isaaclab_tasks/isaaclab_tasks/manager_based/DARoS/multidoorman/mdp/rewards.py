# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    distance = torch.abs(joint_pos - target)
    
    return -distance.sum(dim=-1)

def velocity_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.1
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    vel_magnitude = torch.abs(joint_vel)
    excess_velocity = torch.clamp(vel_magnitude - velocity_threshold, min=0.0)
    
    return -excess_velocity.sum(dim=-1)

def platform_stability_reward(
    env: ManagerBasedRLEnv,
    platform_cfg: SceneEntityCfg,
    stability_threshold: float = 0.1,
) -> torch.Tensor: 
    
    #TODO: Implement platform stability reward so that the platform remains stable during the task.

    return torch.zeros(env.num_envs, device=env.device)

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)

def approach_ee_handle(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    handle_pos = handle_frame.data.target_pos_w[:, 0, :]    
    
    distance = torch.norm(ee_pos - handle_pos, dim=-1)
    
    return torch.exp(-distance / threshold)

def handle_grasp_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]
    
    handle_pos = door.data.joint_pos[:, 1] 
    
    return torch.abs(handle_pos)

def handle_twist_reward(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    door_cfg: SceneEntityCfg,
    target_twist: float = 1.57,
) -> torch.Tensor:

    door: Articulation = env.scene[door_cfg.name]
    
    handle_joint_pos = door.data.joint_pos[:, door_cfg.joint_ids[0]]
    
    twist_progress = torch.abs(handle_joint_pos) / target_twist
    twist_progress = torch.clamp(twist_progress, 0.0, 1.0)
    
    twist_reward = twist_progress * 10.0
    
    completion_threshold = 0.9 * target_twist
    completion_bonus = torch.where(
        torch.abs(handle_joint_pos) >= completion_threshold,
        torch.tensor(5.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    handle_joint_vel = door.data.joint_vel[:, door_cfg.joint_ids[0]]
    velocity_reward = torch.where(
        torch.abs(handle_joint_pos) > 0.1,
        torch.clamp(torch.abs(handle_joint_vel), 0.0, 2.0),
        torch.tensor(0.0, device=env.device)
    )
    
    return twist_reward + completion_bonus + velocity_reward * 0.1

def ee_behind_robot_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    penalty_scale: float = 5.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    
    robot_rot_matrix = matrix_from_quat(robot_quat)
    robot_forward = robot_rot_matrix[:, :, 0]
    
    ee_relative_pos = ee_pos - robot_pos
    
    forward_projection = torch.sum(ee_relative_pos * robot_forward, dim=-1)
    
    penalty = torch.where(
        forward_projection < 0.0,
        forward_projection * penalty_scale,
        torch.tensor(0.0, device=env.device)
    )
    
    return penalty

def ee_handle_alignment_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    alignment_axis: int = 2,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    handle_pos = handle_frame.data.target_pos_w[:, 0, :]
    
    axis_difference = torch.abs(ee_pos[:, alignment_axis] - handle_pos[:, alignment_axis])
    
    return torch.exp(-axis_difference * 10.0)

def ee_handle_orientation_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    weight_factor: float = 2.0,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]
    
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    handle_quat = handle_frame.data.target_quat_w[:, 0, :]
    
    quat_similarity = torch.abs(torch.sum(ee_quat * handle_quat, dim=-1))
    
    return quat_similarity * weight_factor

def progressive_distance_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    close_threshold: float = 0.05,
    far_threshold: float = 0.3,
) -> torch.Tensor:

    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    handle_pos = handle_frame.data.target_pos_w[:, 0, :]

    distance = torch.norm(ee_pos - handle_pos, dim=-1)
    
    very_close_reward = torch.where(
        distance < close_threshold,
        5.0 * (1.0 - distance / close_threshold),
        torch.tensor(0.0, device=env.device)
    )
    
    medium_reward = torch.where(
        (distance >= close_threshold) & (distance < far_threshold),
        2.0 * (1.0 - (distance - close_threshold) / (far_threshold - close_threshold)),
        torch.tensor(0.0, device=env.device)
    )
    
    far_reward = torch.where(
        distance >= far_threshold,
        0.5 * torch.exp(-distance),
        torch.tensor(0.0, device=env.device)
    )
    
    return very_close_reward + medium_reward + far_reward

def handle_approach_velocity_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    velocity_weight: float = 1.0,
) -> torch.Tensor:

    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    handle_pos = handle_frame.data.target_pos_w[:, 0, :]
    
    direction_to_handle = handle_pos - ee_pos
    distance = torch.norm(direction_to_handle, dim=-1, keepdim=True)
    
    direction_normalized = direction_to_handle / (distance + 1e-6)
    
    joint_velocities = robot.data.joint_vel[:, -6:]

    ee_velocity_magnitude = torch.norm(joint_velocities, dim=-1)

    approach_reward = torch.where(
        distance.squeeze() < 0.2,
        ee_velocity_magnitude * velocity_weight,
        torch.tensor(0.0, device=env.device)
    )
    
    return approach_reward

def handle_reach_milestone_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    milestones: list = [0.3, 0.2, 0.1, 0.05],
    milestone_rewards: list = [1.0, 2.0, 5.0, 10.0],
) -> torch.Tensor:

    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    handle_pos = handle_frame.data.target_pos_w[:, 0, :]
    
    distance = torch.norm(ee_pos - handle_pos, dim=-1)

    total_reward = torch.zeros_like(distance)

    for i, (milestone, reward) in enumerate(zip(milestones, milestone_rewards)):
        milestone_achieved = distance < milestone

        total_reward = torch.where(milestone_achieved, reward, total_reward)
    
    return total_reward

def ee_height_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    max_height_above_robot: float = 0.8,
    penalty_scale: float = 5.0,
) -> torch.Tensor:

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    robot_pos = robot.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    
    height_diff = ee_pos[:, 2] - robot_pos[:, 2]
    
    penalty = torch.where(
        height_diff > max_height_above_robot,
        (height_diff - max_height_above_robot) * penalty_scale, 
        torch.tensor(0.0, device=env.device) 
    )
    
    return -penalty

def extreme_joint_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_limits: dict = None,
    penalty_scale: float = 5.0,
) -> torch.Tensor:

    if joint_limits is None:
        joint_limits = {
            "r_joint2": (-1.5, 1.5),    
            "r_joint4": (-1.5, 1.5),    
            "r_joint5": (-1.5, 1.5),    
        }
    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_names = asset_cfg.joint_names if hasattr(asset_cfg, 'joint_names') else []
    
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    
    for joint_name, (min_limit, max_limit) in joint_limits.items():
        try:
            all_joint_names = asset.joint_names
            if joint_name in all_joint_names:
                joint_idx_in_asset = all_joint_names.index(joint_name)

                if joint_idx_in_asset in asset_cfg.joint_ids:
                    joint_idx_in_selection = asset_cfg.joint_ids.index(joint_idx_in_asset)
                    joint_values = joint_pos[:, joint_idx_in_selection]
                    
                    upper_violation = torch.clamp(joint_values - max_limit, min=0.0)

                    lower_violation = torch.clamp(min_limit - joint_values, min=0.0)
                    
                    total_penalty += (upper_violation + lower_violation) * penalty_scale
        except (ValueError, IndexError):
            continue
    
    return -total_penalty

def ee_orientation_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    target_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    orientation_weight: float = 1.0,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    
    ee_quat = ee_frame.data.target_quat_w[:, 0, :] 
    
    target_quat = torch.tensor(target_orientation, device=env.device, dtype=torch.float32)
    target_quat = target_quat.unsqueeze(0).expand(env.num_envs, -1)
    
    quat_similarity = torch.abs(torch.sum(ee_quat * target_quat, dim=-1))
    
    orientation_reward = quat_similarity * orientation_weight
    
    return orientation_reward

def ee_joint_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_joint_angle: float = 0.0,
    joint_name: str = "r_joint6",
    orientation_weight: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    try:
        joint_idx_in_asset = asset.joint_names.index(joint_name)
        if joint_idx_in_asset in asset_cfg.joint_ids:
            joint_idx_in_selection = asset_cfg.joint_ids.index(joint_idx_in_asset)
            current_angle = asset.data.joint_pos[:, asset_cfg.joint_ids[joint_idx_in_selection]]
            
            angle_diff = torch.abs(current_angle - target_joint_angle)
            
            reward = torch.exp(-angle_diff * 5.0) * orientation_weight
            
            return reward
        else:
            return torch.zeros(env.num_envs, device=env.device)
    except (ValueError, IndexError):
        return torch.zeros(env.num_envs, device=env.device)
