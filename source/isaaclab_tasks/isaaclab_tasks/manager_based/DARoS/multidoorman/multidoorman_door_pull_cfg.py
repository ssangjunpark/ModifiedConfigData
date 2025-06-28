import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

DOOR_USD_PATH = "/home/isaac/Documents/Github/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/DARoS/multidoorman/assets/doors/realman_doorpull.usd"

DOOR_CONFIG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=DOOR_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # Remove fix_root_link=True - this is causing the issue
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "door_hinge": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "door_handle": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=5.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)