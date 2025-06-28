import copy
import dataclasses

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
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .multidoorman_env_cfg import MultidoormanEnvCfg
from .multidoorman_robot_cfg import ROBOT_CONFIG
from .multidoorman_door_cfg import DOOR_CONFIG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG

@configclass
class MultidoormanOpenDoorEnvCfg(MultidoormanEnvCfg):
    """Configuration for the MultiDoorMan Open Door environment."""

    def __post_init__(self):
        super().__post_init__()

        
        _robot_cfg = dataclasses.replace(ROBOT_CONFIG, prim_path="{ENV_REGEX_NS}/Robot")
        
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
        
        _robot_cfg.init_state = dataclasses.replace(
            _robot_cfg.init_state, 
            pos=(0.0, 0.7, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),
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

@configclass
class MultidoormanOpenDoorEnvCfgPlay(MultidoormanOpenDoorEnvCfg):
    def __post_init__(self):
        # post init of parent
        from . import mdp
        self.rewards.approach_handle = RewTerm(
            func=mdp.approach_ee_handle,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame", joint_names=["right_end_effector"]),
                "handle_frame_cfg": SceneEntityCfg("door_handle_frame", joint_names=["door_handle"]),
                "threshold": 0.05,
            },
            weight=2.0,
        )
        
        self.rewards.handle_twist = RewTerm(
            func=mdp.handle_twist_reward,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "door_cfg": SceneEntityCfg("door", joint_names=["joint_1", "joint_2"]),
                "target_twist": 1.57, 
            },
            weight=5.0,
        )
        super().__post_init__()
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 60.0