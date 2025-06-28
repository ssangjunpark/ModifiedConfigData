# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import MISSING

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.sensors import TiledCameraCfg

from . import mdp
import random
import os

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip

from .multidoorman_robot_cfg import ROBOT_CONFIG # isort:skip
from .multidoorman_door_cfg import DOOR_CONFIG # isort:skip

@configclass
class MultidoormanSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # door: Will be randomly populated by env cfg
    door: ArticulationCfg = MISSING


    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # cameras
    top_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link/Head",
        update_period=0.1,
        height=256,
        width=256,
        data_types=['rgb'],
        spawn=None,
    )

    left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_link6/Left",
        update_period=0.1,
        height=256,
        width=256,
        data_types=['rgb'],
        spawn=None,
    )
    
    right_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_link6/Right",
        update_period=0.1,
        height=256,
        width=256,
        data_types=['rgb'],
        spawn=None,
    )


##
# MDP settings
##

@configclass
class CurriculumCfg:
    pass
    # curriculum settings
    # 1. Handle grasp
    # 2. Handle manipulation
    # 3. Open Door
    # 4. Door navigation

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "r_joint1",
            "r_joint2",
            "r_joint3",
            "r_joint4",  
            "r_joint5",  
            "r_joint6",
        ],
        scale=0.5,
        use_default_offset=True,
    )

    platform_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["platform_joint"],
        scale=0.5,  # Increased scale for better control authority
        use_default_offset=True,
    )

    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "l_joint1",
            "l_joint2",
            "l_joint3",    
            "l_joint4",
            "l_joint5",
            "l_joint6",
        ],
        scale=0.5,
        use_default_offset=True,
    )

    # right_gripper_action = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "r_gripper_finger1_joint",  # Primary control joint
    #         # Other joints are mimic joints handled automatically by USD
    #     ],
    #     scale=1.0,  # Full scale for gripper control
    #     use_default_offset=True,
    # )

    # left_gripper_action = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "l_gripper_finger1_joint",  # Primary control joint
    #         # Other joints are mimic joints handled automatically by USD
    #     ],
    #     scale=1.0,  # Full scale for gripper control
    #     use_default_offset=True,
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    # Location relative to the door and design the observation space 
    # what we want the robot to see 
    # joint position and velocity
    # location relate to the door
    # know the handle state and the position of the handle
    # door handle position relative to the robot 
    # status of the door handle (the full orientation)
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        right_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"])}
        )
        
        platform_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"])}
        )
        platform_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"])}
        )
        
        left_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6"])}
        )
        
        # right_gripper_joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
        #         "r_gripper_finger1_joint"
        #     ])}
        # )
        
        # left_gripper_joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
        #         "l_gripper_finger1_joint"
        #     ])}
        # )
        
        door_handle_position = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("door", body_names=["link_2"])}
        )
        
        door_handle_orientation = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("door", body_names=["link_2"])}
        )
        
        door_base_position = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("door", body_names=["base"])}
        )
        
        door_opening_angle = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["joint_1"])}
        )
        
        handle_rotation_angle = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["joint_2"])}
        )
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_right_arm_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_left_arm_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6"]),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_platform_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_door_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["joint_1"]),
            "position_range": (0.0, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_handle_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["joint_2"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # reset_right_gripper_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["r_gripper_finger1_joint"]),
    #         "position_range": (0.0, 0.1),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # reset_left_gripper_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["l_gripper_finger1_joint"]),
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    
    # (3) Door opening reward - main objective
    door_opening = RewTerm(
        func=mdp.joint_pos_target,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["joint_1"]),
            "target": 1.0,
        }
    )
    
    # (4) Handle manipulation reward - encourage turning the handle
    handle_manipulation = RewTerm(
        func=mdp.joint_pos_target,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["joint_2"]),
            "target": 0.5,
        }
    )

    # (5) Action penalties for smooth movement
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    
    # (6) Joint velocity penalty for smooth movement 
    right_arm_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"])},
    )

    # (7) Left arm stability (keep left arm still)
    left_arm_stability = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # # (2) Door successfully opened - specify exact joint
    # door_opened = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg("door", joint_names=["joint_1"]),
    #     }
    # )


##
# Environment configuration
##


@configclass
class MultidoormanEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MultidoormanSceneCfg = MultidoormanSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        self.scene.robot = ROBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Add door configuration
        self.scene.door = DOOR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Door")

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link_underpan",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/l_link6",
                    name="right_end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1), 
                    ),
                ),
            ],
        )
        
        self.decimation = 2
        self.episode_length_s = 5.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.viewer.eye = (1.0, 1.0, 0.8)
        self.viewer.lookat = (0.5, 0.0, 0.2)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0


@configclass
class MultidoormanEnvCfg_PLAY(MultidoormanEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3
        # disable randomization for play
        self.observations.policy.enable_corruption = False