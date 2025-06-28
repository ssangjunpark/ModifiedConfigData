from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def ee_handle_distance_threshold(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Termination condition for end-effector reaching close to door handle.

    Args:
        env: The environment.
        ee_frame_cfg: The end-effector frame configuration.
        handle_frame_cfg: The door handle frame configuration.
        threshold: The distance threshold for success. Defaults to 0.05m.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    handle_frame = env.scene[handle_frame_cfg.name]

    # Get positions
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    handle_pos = handle_frame.data.target_pos_w[:, 0, :]

    # Calculate distance
    distance = torch.norm(ee_pos - handle_pos, dim=-1)

    # Return true if within threshold
    return distance < threshold


def robot_workspace_limit(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    bounds: tuple[list[float], list[float]],
) -> torch.Tensor:
    """Termination condition for robot moving outside workspace bounds.

    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        bounds: Tuple of (min_bounds, max_bounds) as lists of [x, y, z].
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # Get robot root position
    robot_pos = robot.data.root_state_w[:, :3]

    min_bounds = torch.tensor(bounds[0], device=env.device)
    max_bounds = torch.tensor(bounds[1], device=env.device)

    # Check if robot is outside bounds
    outside_bounds = torch.any(
        (robot_pos < min_bounds) | (robot_pos > max_bounds),
        dim=-1,
    )

    return outside_bounds

def robot_orientation_limit(
        env: ManagerBasedRLEnv, 
        robot_cfg: SceneEntityCfg,
        orientation_bounds: tuple[list[float], list[float]],
) -> torch.Tensor:
    """Terminates robot if its orientation is outside specified bounds in cfg file."""

    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    robot_quat = robot.data.root_state_w[:, 3:7]

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    robot_euler = torch.atan2(
        2 * (robot_quat[:, 0] * robot_quat[:, 1] + robot_quat[:, 2] * robot_quat[:, 3]),
        1 - 2 * (robot_quat[:, 1]**2 + robot_quat[:, 2]**2),
    )

    # Check if orientation is outside bounds
    min_bounds = torch.tensor(orientation_bounds[0], device=env.device)
    max_bounds = torch.tensor(orientation_bounds[1], device=env.device)

    outside_bounds = torch.any(
        (robot_euler < min_bounds) | (robot_euler > max_bounds),
        dim=-1,
    )

    return outside_bounds