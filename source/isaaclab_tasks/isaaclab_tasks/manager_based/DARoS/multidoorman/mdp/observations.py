from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def door_handle_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["link_2"]),
) -> torch.Tensor:
    """The position of the door handle in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]
    
    # Get handle position (link_2)
    handle_pos_w = door.data.body_state_w[:, door_cfg.body_ids[0], :3]
    handle_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], handle_pos_w
    )
    return handle_pos_b

def door_handle_orientation_w(
    env: ManagerBasedRLEnv,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["link_2"]),
) -> torch.Tensor:
    """The orientation of the door handle in world frame."""
    door: Articulation = env.scene[door_cfg.name]
    return door.data.body_state_w[:, door_cfg.body_ids[0], 3:7]

def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b

def ee_to_handle_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["link_2"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Distance between end-effector and door handle."""
    door: Articulation = env.scene[door_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get positions
    handle_pos_w = door.data.body_state_w[:, door_cfg.body_ids[0], :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    
    # Calculate distance
    distance = torch.norm(ee_pos_w - handle_pos_w, dim=-1, keepdim=True)
    return distance