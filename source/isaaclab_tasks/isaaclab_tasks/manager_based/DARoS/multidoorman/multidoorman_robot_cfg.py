import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

REALMAN_USD_PATH = "/home/isaac/Documents/Github/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/DARoS/multidoorman/assets/realman/urdf/rm_dual_65.usd"

ROBOT_CONFIG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=REALMAN_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=4,
            #fix_root_link=False,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "platform_joint": 0.0,
            "head_joint1": 0.0,
            "head_joint2": 0.0,
            "r_joint1": 0.0,
            "r_joint2": math.pi/2 + math.pi/4,
            "r_joint3": 0.0,
            "r_joint4": 0.0,
            "r_joint5": 0.0,
            "r_joint6": 0.0,
            "l_joint1": 0.0,
            "l_joint2": math.pi/2 + math.pi/4,
            "l_joint3": 0.0,
            "l_joint4": 0.0,
            "l_joint5": 0.0,
            "l_joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "platform_joint": ImplicitActuatorCfg(
            joint_names_expr=["platform_joint"],
            stiffness=1000.0,
            damping=200.0,
            effort_limit=500.0,
            velocity_limit=0.5,
        ),
        "head_joint1": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1"],
            stiffness=150.0,
            damping=80.0,
            effort_limit=100.0,
            velocity_limit=1.0,
        ),
        "head_joint2": ImplicitActuatorCfg(
            joint_names_expr=["head_joint2"],
            stiffness=150.0,
            damping=80.0,
            effort_limit=100.0,
            velocity_limit=1.0,
        ),
        
        "r_joint1": ImplicitActuatorCfg(
            joint_names_expr=["r_joint1"],
            stiffness=250.0,
            damping=120.0,
            effort_limit=300.0,
            velocity_limit=0.8,
        ),
        "r_joint2": ImplicitActuatorCfg(
            joint_names_expr=["r_joint2"],
            stiffness=250.0,
            damping=120.0,
            effort_limit=300.0,
            velocity_limit=0.8,
        ),
        "r_joint3": ImplicitActuatorCfg(
            joint_names_expr=["r_joint3"],
            stiffness=200.0,
            damping=100.0,
            effort_limit=250.0,
            velocity_limit=1.0,
        ),
        "r_joint4": ImplicitActuatorCfg(
            joint_names_expr=["r_joint4"],
            stiffness=150.0,
            damping=80.0,
            effort_limit=150.0,
            velocity_limit=1.2,
        ),
        "r_joint5": ImplicitActuatorCfg(
            joint_names_expr=["r_joint5"],
            stiffness=150.0,
            damping=80.0,
            effort_limit=150.0,
            velocity_limit=1.2,
        ),
        "r_joint6": ImplicitActuatorCfg(
            joint_names_expr=["r_joint6"],
            stiffness=100.0,
            damping=60.0,
            effort_limit=100.0,
            velocity_limit=1.5,
        ),
        
        "l_joint1": ImplicitActuatorCfg(
            joint_names_expr=["l_joint1"],
            stiffness=250.0,
            damping=120.0,
            effort_limit=300.0,
            velocity_limit=0.8,
        ),
        "l_joint2": ImplicitActuatorCfg(
            joint_names_expr=["l_joint2"],
            stiffness=250.0,
            damping=120.0,
            effort_limit=300.0,
            velocity_limit=0.8,
        ),
        "l_joint3": ImplicitActuatorCfg(
            joint_names_expr=["l_joint3"],
            stiffness=200.0,
            damping=100.0,
            effort_limit=250.0,
            velocity_limit=1.0,
        ),
        "l_joint4": ImplicitActuatorCfg(
            joint_names_expr=["l_joint4"],
            stiffness=150.0,
            damping=80.0,
            effort_limit=150.0,
            velocity_limit=1.2,
        ),
        "l_joint5": ImplicitActuatorCfg(
            joint_names_expr=["l_joint5"],
            stiffness=150.0,
            damping=80.0,
            effort_limit=150.0,
            velocity_limit=1.2,
        ),
        "l_joint6": ImplicitActuatorCfg(
            joint_names_expr=["l_joint6"],
            stiffness=100.0,
            damping=60.0,
            effort_limit=100.0,
            velocity_limit=1.5,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)