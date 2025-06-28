import copy
import dataclasses
import math

from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.managers import EventTermCfg as EventTerm

from .multidoorman_env_cfg import MultidoormanEnvCfg, ActionsCfg, CurriculumCfg
from .multidoorman_robot_cfg import ROBOT_CONFIG
from .multidoorman_door_cfg import DOOR_CONFIG
from . import mdp
from isaaclab.markers.config import FRAME_MARKER_CFG

@configclass
class GraspActionsCfg:
    """Custom action specifications for grasp environment - using left arm instead of right arm."""

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

    platform_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["platform_joint"],
        scale=0.5,
        use_default_offset=True,
    )

@configclass
class MultidoormanGraspCfg(MultidoormanEnvCfg):
    """Configuration for the MultiDoorMan Grasp environment."""

    def __post_init__(self):
        super().__post_init__()

        self.actions = GraspActionsCfg()
        
        _robot_cfg = dataclasses.replace(ROBOT_CONFIG, prim_path="{ENV_REGEX_NS}/Robot")
        
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()

        _new_actuators = {}
        
        for actuator_name, actuator_cfg in _robot_cfg.actuators.items():
            if actuator_name in ["platform_joint", "head_joint1", "head_joint2"] or actuator_name.startswith("l_joint"):
                _new_actuators[actuator_name] = actuator_cfg

        _locked_right_arm_actuators = {
            joint: ImplicitActuatorCfg(
                joint_names_expr=[joint],
                stiffness=100000.0,
                damping=10000.0,
                effort_limit_sim=2.0,
                velocity_limit_sim=0.001,
            )
            for joint in ["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"]
        }

        _new_actuators.update(_locked_right_arm_actuators)
        
        if "platform_joint" in _new_actuators:
            _new_actuators["platform_joint"] = ImplicitActuatorCfg(
                joint_names_expr=["platform_joint"],
                stiffness=1000.0,
                damping=200.0,
                effort_limit_sim=500.0,
                velocity_limit_sim=0.5,
            )
            
        _robot_cfg.actuators = _new_actuators
        
        _robot_cfg.init_state = dataclasses.replace(
            _robot_cfg.init_state, 
            pos=(0.0, 0.6, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "platform_joint": 0.3,
                "head_joint1": 0.0,
                "head_joint2": 0.0,
                
                # Right arm joints
                "r_joint1": 0.0,
                "r_joint2": math.pi/2 + math.pi/6,
                "r_joint3": 0.0,
                "r_joint4": 0.0,
                "r_joint5": 0.0,
                "r_joint6": 0.0,

                # Left arm joints
                "l_joint1": 0.0,
                "l_joint2": -math.pi/2 - math.pi/6,
                "l_joint3": 0.0,
                "l_joint4": -math.pi/2,
                "l_joint5": -math.pi/2,
                "l_joint6": 0.0,
                "r_gripper_finger1_joint": 0.0,
                "l_gripper_finger1_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        )
        self.scene.robot = _robot_cfg
        
        _door_cfg = dataclasses.replace(DOOR_CONFIG, prim_path="{ENV_REGEX_NS}/Door")
        
        if _door_cfg.init_state is None:
            _door_cfg.init_state = ArticulationCfg.InitialStateCfg()
        
        _door_cfg.init_state = dataclasses.replace(
            _door_cfg.init_state,
            pos=(0.0, 1.3, 1.0),
            rot=(-0.707, 0.0, 0.0, 0.707),
        )

        self.scene.door = _door_cfg

        marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        }
        marker_cfg.prim_path = "/Visuals/LeftArmEndEffector" 
        
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link_underpan",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/l_link6",
                    name="left_end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, -0.1),
                        rot=(0.0, 0.0, 1.0, 0.0),
                    ),
                ),
            ],
        )

        door_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        door_marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        }
        door_marker_cfg.prim_path = "/Visuals/DoorHandleMarker"
        
        # import random



        self.scene.door_handle_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Door/base",
            debug_vis=True,
            visualizer_cfg=door_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Door/link_2",
                    name="door_handle",
                    offset=OffsetCfg(
                        pos=(0.0,0.0, -0.1),
                        rot=(0.0, 0.0, 1.0, 0.0),
                    ),
                ),
            ],
        )

        self.observations.policy.ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        
        self.observations.policy.handle_pos = ObsTerm(
            func=mdp.door_handle_position_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "door_cfg": SceneEntityCfg("door", body_names=["link_2"]),
            },
        )
        
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint.*"])},
        )

        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint.*"])},
        )
        
        self.observations.policy.handle_orientation = ObsTerm(
            func=mdp.door_handle_orientation_w,
            params={"door_cfg": SceneEntityCfg("door", body_names=["link_2"])},
        )
        
        self.observations.policy.ee_handle_distance = ObsTerm(
            func=mdp.ee_to_handle_distance,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "door_cfg": SceneEntityCfg("door", body_names=["link_2"]),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        
        self.observations.policy.platform_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"])},
        )
        
        self.observations.policy.platform_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"])},
        )
        
        self.rewards.approach_handle = RewTerm(
            func=mdp.approach_ee_handle,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "threshold": 0.05,
            },
            weight=1.5,
        )
        
        self.rewards.progressive_distance = RewTerm(
            func=mdp.progressive_distance_reward,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "close_threshold": 0.03,
                "far_threshold": 0.25,
            },
            weight=1.0,
        )
        
        self.rewards.vertical_alignment = RewTerm(
            func=mdp.ee_handle_alignment_reward,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "alignment_axis": 2,
            },
            weight=1.0,
        )
        
        self.rewards.horizontal_alignment = RewTerm(
            func=mdp.ee_handle_alignment_reward,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "alignment_axis": 1,
            },
            weight=1.0,
        )

        self.rewards.ee_orientation = RewTerm(
            func=mdp.ee_joint_orientation_reward,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint6"]),
                "target_joint_angle": 0.0,
                "joint_name": "l_joint6",
                "orientation_weight": 2.0,
            },
            weight=2.0,
        )
        
        self.rewards.approach_velocity = RewTerm(
            func=mdp.handle_approach_velocity_reward,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
                "velocity_weight": 0.5,
            },
            weight=0.5,
        )
        
        self.rewards.reach_milestones = RewTerm(
            func=mdp.handle_reach_milestone_reward,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "milestones": [0.25, 0.15, 0.08, 0.04],
                "milestone_rewards": [0.5, 1.0, 2.0, 5.0],
            },
            weight=0.2,
        )
        
        self.rewards.joint_smoothness = RewTerm(
            func=mdp.joint_vel_l2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint.*"])},
            weight=-0.1,
        )
        
        self.rewards.natural_joint_positions = RewTerm(
            func=mdp.joint_pos_target,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint2", "l_joint4", "l_joint5"]),
                "target": 0.0,
            },
            weight=-0.05,
        )
        
        self.rewards.ee_behind_penalty = RewTerm(
            func=mdp.ee_behind_robot_penalty,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "penalty_scale": 10.0,
            },
            weight=3.0,
        )
        
        self.rewards.height_alignment = RewTerm(
            func=mdp.ee_handle_alignment_reward,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "alignment_axis": 2,
            },
            weight=1.5,
        )

        self.terminations.success = DoneTerm(
            func=mdp.ee_handle_distance_threshold,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "threshold": 0.08,
            },
        )

        self.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)

        self.rewards.extreme_joint_penalty = RewTerm(
            func=mdp.extreme_joint_penalty,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint.*"]),
                "joint_limits": {
                    "l_joint2": (-1.4, 1.4),
                    "l_joint4": (-1.4, 1.4), 
                    "l_joint5": (-1.4, 1.4),
                },
                "penalty_scale": 3.0,
            },
            weight=1.5,
        )
        
        self.events = GraspEventsCfg()
        
@configclass
class MultidoormanGraspCfgPlay(MultidoormanGraspCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 20.0

@configclass
class MultidoormanGraspCfgTrain(MultidoormanGraspCfg):
    """Training-specific configuration with optimized parameters."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = True
        self.episode_length_s = 30.0
        
        self.curriculum = CurriculumCfg()


@configclass
class MultidoormanGraspCfgLog(MultidoormanGraspCfg):
    """Training-specific configuration with optimized parameters."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        #self.episode_length_s = 2
        self.episode_length_s = 1
        
        self.curriculum = CurriculumCfg()

@configclass
class GraspEventsCfg:
    """Custom event specifications for grasp environment with robot Y position randomization."""

    reset_platform_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"]),
            "position_range": (0.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_head_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["head_joint1", "head_joint2"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_right_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_left_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_gripper_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["r_gripper_finger1_joint", "l_gripper_finger1_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


    reset_handle_position = EventTerm(
        func=mdp.reset_handle_with_custom_offset,
        mode="reset",
        params={
            "door_cfg": SceneEntityCfg("door"),
        },
    )

    reset_robot_y_position = EventTerm(
        func=mdp.reset_robot_y_position,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_y_range": (0.5, 0.7),
        },
    )