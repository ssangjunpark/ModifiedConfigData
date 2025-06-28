# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import random
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import reset_joints_by_offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_robot_y_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_y_range: tuple[float, float] = (0.4, 0.8),
    robot_x: float = 0.0,
    robot_z: float = 0.25,
) -> None:
    """Reset robot Y position randomly while keeping environments separate."""
    robot: Articulation = env.scene[robot_cfg.name]
    
    num_resets = len(env_ids)
    robot_y_positions = torch.FloatTensor(num_resets).uniform_(
        robot_y_range[0], robot_y_range[1]
    ).to(env.device)
    
    current_robot_pos = robot.data.root_pos_w[env_ids].clone()
    current_robot_quat = robot.data.root_quat_w[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    
    current_robot_pos[:, 1] = env_origins[:, 1] + robot_y_positions
    current_robot_pos[:, 0] = env_origins[:, 0] + robot_x
    current_robot_pos[:, 2] = env_origins[:, 2] + robot_z
    
    root_state = torch.cat([current_robot_pos, current_robot_quat], dim=-1)
    
    robot.write_root_pose_to_sim(
        root_pose=root_state,
        env_ids=env_ids
    )

def reset_handle_with_custom_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> None:
    """Reset door handle (joint_2) using custom random offset pattern."""
    import random
    
    door: Articulation = env.scene[door_cfg.name]
    
    joint_ids, joint_names = door.find_joints(["joint_2"])
    
    if len(joint_ids) == 0:
        return
    
    if len(joint_ids) > 1:
        joint_ids = joint_ids[0]
        joint_names = joint_names[0]
    else:
        joint_ids = joint_ids[0]
        joint_names = joint_names[0]
    
    joint_offsets = []
    for _ in range(len(env_ids)):
        x_offset = random.uniform(0.0, 0.0)
        y_offset = random.uniform(0.0, 0.2)
        z_offset = random.uniform(-0.2, 0.2)
        
        joint_offsets.append(y_offset)
    
    joint_offsets_tensor = torch.tensor(joint_offsets, device=env.device).unsqueeze(1)
    
    default_joint_pos = door.data.default_joint_pos[env_ids, joint_ids].unsqueeze(1)
    
    new_joint_pos = default_joint_pos + joint_offsets_tensor
    new_joint_vel = torch.zeros_like(new_joint_pos)
    
    door.write_joint_state_to_sim(
        position=new_joint_pos,
        velocity=new_joint_vel,
        joint_ids=[joint_ids],
        env_ids=env_ids,
    )

def reset_door_handle_origin_randomization(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> None:
    """Reset door handle by using pre-generated URDF variants with randomized joint_2 origins."""
    reset_joints_by_offset(
        env, 
        env_ids, 
        asset_cfg=door_cfg, 
        position_range=(-0.1, 0.1), 
        velocity_range=(0.0, 0.0)
    )


