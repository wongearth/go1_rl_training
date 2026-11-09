# save as check_order.py, run with your isaac python
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG as C
print("init_state.joint_pos:")
for k, v in C.init_state.joint_pos.items():
    print(f"  {k}: {v}")
print("\nactuators:")
for k, v in C.actuators.items():
    print(f"  group={k}  joints={v.joint_names_expr}  kp={v.stiffness}  kd={v.damping}  eff={v.effort_limit}")