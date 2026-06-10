import argparse
from isaaclab.app import AppLauncher

# 1. BOOT UP OMNIVERSE FIRST
# This must happen before any Isaac Lab or Omni imports
parser = argparse.ArgumentParser(description="List registered environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. NOW IT IS SAFE TO IMPORT TASKS
import gymnasium as gym
import isaaclab_tasks 

# IMPORT YOUR CUSTOM MODULE HERE! 
# If your folder is named differently, change this to match.
try:
    import go1_tasks
except ImportError:
    print("Warning: Could not import 'go1_tasks'. Check your module name.")

# 3. FETCH AND PRINT ENVIRONMENTS
print("\n" + "="*40)
print("REGISTERED GO1 ENVIRONMENTS:")
print("="*40)

all_envs = gym.envs.registry.keys()
go1_envs = [env for env in all_envs if "Go1" in env]

if not go1_envs:
    print("No environments containing 'Go1' were found in the registry.")
else:
    for env in sorted(go1_envs):
        print(f" - {env}")

print("="*40 + "\n")

# 4. CLOSE THE APP SAFELY
simulation_app.close()