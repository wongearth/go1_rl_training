import numpy as np, mujoco, mujoco.viewer, time
XML = ".././mujoco_menagerie/unitree_go1/scene.xml"
ORDER = ['FL_hip_joint','FR_hip_joint','RL_hip_joint','RR_hip_joint',
         'FL_thigh_joint','FR_thigh_joint','RL_thigh_joint','RR_thigh_joint',
         'FL_calf_joint','FR_calf_joint','RL_calf_joint','RR_calf_joint']
QDEF = np.array([0.1,-0.1,0.1,-0.1, 0.7,0.7,0.9,0.9, -1.3,-1.3,-1.3,-1.3])
KP, KD = 40.0, 1.0

m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005
d = mujoco.MjData(m)

print("nu:", m.nu)
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trntype = m.actuator_trntype[i]   # 0 = joint (motor/position), transmission target
    tgt = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i,0])
    print(f"{i}: actuator='{name}'  ->  joint='{tgt}'")

    # gaintype: 0=fixed(motor/torque), others imply position/velocity servo
    # biastype: 0=none(motor), 1=affine(position servo)
    print(f"{name:10s}  gaintype={m.actuator_gaintype[i]}  biastype={m.actuator_biastype[i]}"
          f"  gainprm0={m.actuator_gainprm[i,0]:.1f}  biasprm1={m.actuator_biasprm[i,1]:.1f}"
          f"  ctrlrange={m.actuator_ctrlrange[i]}  forcerange={m.actuator_forcerange[i]}")



qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER]
vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER]
aid  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ORDER]
print("actuator ids (any -1 == name mismatch):", aid)

d.qpos[0:3] = [0,0,0.35]; d.qpos[3:7] = [1,0,0,0]
for i,a in enumerate(qadr): d.qpos[a] = QDEF[i]
mujoco.mj_forward(m,d)

with mujoco.viewer.launch_passive(m,d) as v:
    step = 0
    while v.is_running():
        q  = d.qpos[qadr]; qd = d.qvel[vadr]
        tau = np.clip(KP*(QDEF-q) - KD*qd, -23.5, 23.5)
        for i,a in enumerate(aid): d.ctrl[a] = tau[i]
        mujoco.mj_step(m,d); v.sync()
        step += 1
        if step % 100 == 0: print(f"height={d.qpos[2]:.3f}  upright_z={d.qpos[6]:.2f}")
        time.sleep(0.005)