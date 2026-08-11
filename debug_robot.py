from isaaclab.app import AppLauncher
import sys
import os

# Ensure source path is found
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "source"))

# 1. Start the app
app_launcher = AppLauncher()
app = app_launcher.app

from isaaclab.assets import Articulation
from mc_tasks.data.robots.mc_quadruped.mc_quadruped_cfg import MC_QUADRUPED_CFG

def main():
    # 2. Configure and load
    # We remove the prim_path entirely for the load phase to prevent "already exists" errors
    robot_cfg = MC_QUADRUPED_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    
    # 3. Access body names directly from the configuration metadata
    # The 'body_names' property is defined in the ArticulationCfg and parsed from the USD
    print("\n" + "="*40)
    print("DEBUG: ASSET BODY NAMES FROM CONFIG:")
    print("="*40)
    # The robot asset parses the USD structure upon instantiation
    print(robot.body_names) 
    print("="*40)
    
    # If this fails, we know it's a USD path/parse issue.
    # If this succeeds, you have your list!

if __name__ == "__main__":
    main()