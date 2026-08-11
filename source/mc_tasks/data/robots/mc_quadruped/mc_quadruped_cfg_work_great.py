from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.sensors import ContactSensorCfg

import os

# Get the path to the directory where this config file is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(CURRENT_DIR, "quadruped_robot_01.usd")
# USD_PATH = os.path.join(CURRENT_DIR, "quadruped_robot_01_cleaned.usd")

MC_QUADRUPED_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=USD_PATH, # Update this path
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35), # Adjust height if needed
        # joint_pos={".*": 0.0},
        joint_pos={
                "q0": 0.1, "q3": -0.1, "q6": -0.1, "q9": 0.1,        # hips
                 "q1": 0.7, "q4": -0.7, "q7": 0.7, "q10": -0.7,        # upper legs
                 "q2": 1.4, "q5": 1.4, "q8": 1.4, "q11": 1.4,    # lower legs
        },
    ),

    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*"], # Use regex to cover all leg joints
            effort_limit=20.0,       # Update based on your robot's specs
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.0,
            saturation_effort=20.0,
            # effort_limit=40.0,       # Update based on your robot's specs
            # velocity_limit=40.0,
            # stiffness=80.0,
            # damping=1.0,
            # saturation_effort=40.0,
        ),
    },
)
