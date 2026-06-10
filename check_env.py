import gymnasium as gym
import go1_tasks.manager_based.locomotion.move_p2p

print("\n" + "="*50)
envs = [e for e in gym.envs.registry.keys() if "Go1" in e]
print(f"Registered Go1 Environments:\n{envs}")
print("="*50 + "\n")
