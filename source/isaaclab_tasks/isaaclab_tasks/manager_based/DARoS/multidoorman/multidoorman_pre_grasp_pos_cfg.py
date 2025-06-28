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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs.mdp import *  # Import standard mdp functions
from . import mdp  # Import our custom mdp functions

from .multidoorman_env_cfg import MultidoormanEnvCfg
from .multidoorman_robot_cfg import ROBOT_CONFIG
from .multidoorman_door_cfg import DOOR_CONFIG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG

@configclass
class MultidoormanPreGraspPosEnvCfg(MultidoormanEnvCfg):
    """Configuration for the MultiDoorMan Open Door environment."""

    def __post_init__(self):
        super().__post_init__()

        
        _robot_cfg = dataclasses.replace(ROBOT_CONFIG, prim_path="{ENV_REGEX_NS}/Robot")
        
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
        
        _robot_cfg.init_state = dataclasses.replace(
            _robot_cfg.init_state, 
            pos=(0.0, 0.6, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "platform_joint": 0.0,
                "head_joint1": 0.0,
                "head_joint2": 0.0,
                "r_joint1": 0.0,
                "r_joint2": 0.0,
                "r_joint3": 0.0,
                "r_joint4": 0.0,
                "r_joint5": 0.0,
                "r_joint6": 0.0,
                "l_joint1": 0.0,
                "l_joint2": 0.0,
                "l_joint3": 0.0,
                "l_joint4": 0.0,
                "l_joint5": 0.0,
                "l_joint6": 0.0,
                # "r_gripper_finger1_joint": 0.6524,
                # "l_gripper_finger1_joint": 0.6524,
            },
        )
        self.scene.robot = _robot_cfg
        
        _door_cfg = dataclasses.replace(DOOR_CONFIG, prim_path="{ENV_REGEX_NS}/Door")
        
        if _door_cfg.init_state is None:
            _door_cfg.init_state = ArticulationCfg.InitialStateCfg()
        
        _door_cfg.init_state = dataclasses.replace(
            _door_cfg.init_state,
            pos=(0.0, 1.5, 1.0),
            rot=(0.707, 0.0, 0.0, 0.707),
        )

        self.scene.door = _door_cfg

        marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.03, 0.03, 0.03),
            )
        }
        marker_cfg.prim_path = "/Visuals/RightArmEndEffector"
        
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link_underpan",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/r_link6",
                    name="right_end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1),
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
        
        self.scene.door_handle_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Door/base",
            debug_vis=False,
            visualizer_cfg=door_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Door/link_2",
                    name="door_handle",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
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
        
        # Door handle position relative to robot base
        self.observations.policy.handle_pos = ObsTerm(
            func=mdp.door_handle_position_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "door_cfg": SceneEntityCfg("door", body_names=["link_2"]),
            },
        )
        
        # Robot joint positions (right arm only for pre-grasp)
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint.*"])},
        )

        # Robot joint velocities
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint.*"])},
        )
        
        # Door handle orientation
        self.observations.policy.handle_orientation = ObsTerm(
            func=mdp.door_handle_orientation_w,
            params={"door_cfg": SceneEntityCfg("door", body_names=["link_2"])},
        )
        # Reward for approaching the handle
        self.rewards.approach_handle = RewTerm(
            func=mdp.approach_ee_handle,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "threshold": 0.2,
            },
            weight=3.0,
        )
        
        # Penalty for large joint movements (smooth motion)
        self.rewards.joint_smoothness = RewTerm(
            func=mdp.joint_vel_l2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["r_joint.*"])},
            weight=-0.1,
        )

        # Success: End effector reaches close to handle
        self.terminations.success = DoneTerm(
            func=mdp.ee_handle_distance_threshold,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame"),
                "threshold": 0.15,
            },
        )

@configclass
class MultidoormanPreGraspPosEnvCfgPlay(MultidoormanPreGraspPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from . import mdp
        
        # Play-specific environment settings
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 60.0