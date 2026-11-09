"""Minimal standing test with diagnostics -- NO policy.
Run:  python test_stand_clean.py --xml /path/to/scene.xml
"""
import argparse, time
import numpy as np, mujoco, mujoco.viewer

ORDER = ['FL_hip_joint','FR_hip_joint','RL_hip_joint','RR_hip_joint',
         'FL_thigh_joint','FR_thigh_joint','RL_thigh_joint','RR_thigh_joint',
         'FL_calf_joint','FR_calf_joint','RL_calf_joint','RR_calf_joint']
QDEF = np.array([0.1,-0.1,0.1,-0.1, 0.7,0.7,0.9,0.9, -1.3,-1.3,-1.3,-1.3])

p = argparse.ArgumentParser(); p.add_argument("--xml", required=True)
p.add_argument("--height", type=float, default=0.40)
args = p.parse_args()

m = mujoco.MjModel.from_xml_path(args.xml)
m.opt.timestep = 0.005
d = mujoco.MjData(m)

qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER]
name2act = {}
for a in range(m.nu):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
    name2act[jn] = a
aid = [name2act[n] for n in ORDER]

# ---- DIAGNOSTICS ----
print("\n=== actuator kp (want all 40.0 after your edit) ===")
print("kp:", [round(float(m.actuator_gainprm[a, 0]), 1) for a in range(m.nu)])
print("\n=== per-joint mapping / range check ===")
for i, n in enumerate(ORDER):
    lo, hi = m.actuator_ctrlrange[aid[i]]
    flag = 'OUT OF RANGE!' if not (lo <= QDEF[i] <= hi) else ''
    print(f"{n:18s} -> actuator {aid[i]:2d}  target={QDEF[i]:+.2f}  "
          f"range=[{lo:+.2f},{hi:+.2f}]  {flag}")
print()

d.qpos[0:3] = [0, 0, args.height]
d.qpos[3:7] = [1, 0, 0, 0]
for i, a in enumerate(qadr):
    d.qpos[a] = QDEF[i]
for i, a in enumerate(aid):
    d.ctrl[a] = QDEF[i]
mujoco.mj_forward(m, d)

with mujoco.viewer.launch_passive(m, d) as v:
    step = 0
    while v.is_running():
        for i, a in enumerate(aid):
            d.ctrl[a] = QDEF[i]
        mujoco.mj_step(m, d)
        v.sync()
        step += 1
        if step % 100 == 0:
            print(f"height={d.qpos[2]:.3f}  quat_w={d.qpos[3]:.2f}")
        time.sleep(0.005)
