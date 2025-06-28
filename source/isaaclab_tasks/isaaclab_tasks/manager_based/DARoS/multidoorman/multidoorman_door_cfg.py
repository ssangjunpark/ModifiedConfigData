import math
import random
import os
import shutil

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

DOOR_URDF_PATH = "/home/isaac/Documents/Github/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/DARoS/multidoorman/assets/doors/8897/mobility_fixed.urdf"

def create_door_variants():
    """Create multiple URDF files with different handle positions using generate_random_offset pattern."""
    base_urdf = DOOR_URDF_PATH
    variants_dir = "/home/isaac/Documents/Github/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/DARoS/multidoorman/assets/doors/variants/"

    # Original joint_2 position
    orig_x = 0.6142787515093229
    orig_y = -0.8737701913054894
    orig_z = 0.05812568805968965

    # Generate handle positions using your generate_random_offset pattern
    handle_positions = []
    
    # Create multiple variants with your random pattern
    for i in range(20):  # Create 20 variants
        # Use your generate_random_offset pattern
        x_offset = random.uniform(0.0, 0.0)      # X offset (always 0)
        y_offset = random.uniform(0.0, 0.2)      # Y offset (0 to 0.2)
        z_offset = random.uniform(-0.2, 0.2)     # Z offset (-0.2 to 0.2)
        
        new_x = orig_x + x_offset
        new_y = orig_y + y_offset  
        new_z = orig_z + z_offset
        
        handle_positions.append((new_x, new_y, new_z))
    
    variant_paths = []
    
    # Ensure variants directory exists
    os.makedirs(variants_dir, exist_ok=True)
    
    for i, (x, y, z) in enumerate(handle_positions):
        variant_path = f"{variants_dir}/door_variant_{i}.urdf"
        
        try:
            # Read base URDF
            with open(base_urdf, 'r') as f:
                urdf_content = f.read()
            
            # Replace joint_2 origin
            import re
            old_origin = r'<origin xyz="0\.6142787515093229 -0\.8737701913054894 0\.05812568805968965"/>'
            new_origin = f'<origin xyz="{x} {y} {z}"/>'
            
            # Find and replace joint_2 origin specifically
            pattern = r'(<joint name="joint_2"[^>]*>.*?)<origin xyz="[^"]*"/>'
            replacement = f'\\1<origin xyz="{x} {y} {z}"/>'
            urdf_content = re.sub(pattern, replacement, urdf_content, flags=re.DOTALL)
            
            # Write variant
            with open(variant_path, 'w') as f:
                f.write(urdf_content)
            
            variant_paths.append(variant_path)
            print(f"Created door variant {i} with handle at ({x:.3f}, {y:.3f}, {z:.3f})")
            
        except Exception as e:
            print(f"Error creating variant {i}: {e}")
            # If variant creation fails, use original URDF
            variant_paths.append(base_urdf)
    
    return variant_paths

# Get random door variant
def get_random_door_config():
    """Get a random door configuration with different handle position."""
    try:
        variant_paths = create_door_variants()
        selected_path = random.choice(variant_paths)
        print(f"Selected door variant: {selected_path}")
        
        return ArticulationCfg(
            spawn=UrdfFileCfg(
                asset_path=selected_path,
                activate_contact_sensors=False,
                fix_base=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=200.0,
                        damping=50.0,
                    ),
                ),
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
    except Exception as e:
        print(f"Error creating random door config: {e}")
        return DOOR_CONFIG

DOOR_CONFIG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=DOOR_URDF_PATH,
        activate_contact_sensors=False,
        fix_base=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=200.0,
                damping=50.0,
            ),
        ),
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