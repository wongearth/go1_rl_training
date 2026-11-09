"""
Sim2sim playback of the P2P-timed Go1 policy in MuJoCo (mujoco_menagerie unitree_go1).

Loads the JIT policy exported by your Isaac play script
(logs/.../exported/policy.pt) and drives the Go1 with the SAME observation
layout, action scaling, PD gains and control rate used in Isaac Lab training.

Run:
    pip install mujoco
    python play_mujoco_p2p.py \
        --policy /path/to/exported/policy.pt \
        --xml    /path/to/mujoco_menagerie/unitree_go1/scene.xml
"""

import argparse
import math
import time

import numpy as np
import mujoco
import mujoco.viewer
import torch

# =========================================================
# CONFIG -- taken directly from your Isaac env cfg. VERIFY the PD gains
# and q_default against your local UNITREE_GO1_CFG (snippet at bottom of chat).
# =========================================================
ACTION_SCALE = 0.30          # Go1ActionsCfg.joint_pos.scale
DECIMATION   = 4             # policy at 50 Hz
SIM_DT       = 0.005         # 200 Hz  -> control_dt = 0.02
KP           = 40.0          # base_legs actuator stiffness (from Isaac)
KD           = 1.0           # base_legs actuator damping (from Isaac)
EFFORT_LIMIT = 23.5          # Nm, torque clamp (Go1 default)

# Isaac articulation DOF order for Go1 (FL, FR, RL, RR) x (hip, thigh, calf).
# This is the order the POLICY expects for joint_pos / joint_vel / actions.
ISAAC_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
# NOTE: Isaac sorts joints by regex match order which for Go1 groups all hips,
# then all thighs, then all calfs. If your policy limps, the most likely cause
# is this ordering -- print `robot.joint_names` in an Isaac session to confirm
# and reorder this list to match EXACTLY.

# default joint angles (rad), in ISAAC_JOINT_ORDER above.
# Read directly from Isaac robot.data.default_joint_pos -- exact match.
Q_DEFAULT = np.array([
    0.1, -0.1, 0.1, -0.1,        # hips   (FL FR RL RR)
    0.7,  0.7,  0.9,  0.9,       # thighs (front 0.7, rear 0.9)
    -1.3, -1.3, -1.3, -1.3,      # calfs
], dtype=np.float64)

# P2P command ranges (LocalPoseTimeCommandCfg.Ranges)
CMD_X   = (-1.5, 1.5)
CMD_Y   = (-1.5, 1.5)
CMD_YAW = (-3.14, 3.14)
CMD_T   = (1.2, 2.5)


def quat_rotate_inverse(q_wxyz, v):
    """Rotate v by inverse of quaternion q (w,x,y,z). Matches Isaac convention."""
    w, x, y, z = q_wxyz
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * 2.0 * w
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def yaw_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class Go1Runner:
    def __init__(self, xml_path, policy_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

        # Build a joint-name -> actuator-id map by asking each actuator which
        # joint it drives. This is robust to BOTH the menagerie naming
        # (actuator 'FR_hip' drives joint 'FR_hip_joint') AND its per-leg
        # ordering, which differs from Isaac's grouped ordering.
        joint_to_act = {}
        for aid in range(self.model.nu):
            jid_driven = self.model.actuator_trnid[aid, 0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid_driven)
            joint_to_act[jname] = aid

        # build index maps between MuJoCo qpos/qvel and Isaac joint order
        self.mj_jnt_qposadr = []
        self.mj_jnt_qveladr = []
        self.mj_act_id = []
        for name in ISAAC_JOINT_ORDER:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint '{name}' not in XML -- check names/order")
            self.mj_jnt_qposadr.append(self.model.jnt_qposadr[jid])
            self.mj_jnt_qveladr.append(self.model.jnt_dofadr[jid])
            if name not in joint_to_act:
                raise RuntimeError(f"no actuator drives joint '{name}'")
            self.mj_act_id.append(joint_to_act[name])   # actuator that drives this Isaac joint
        self.mj_jnt_qposadr = np.array(self.mj_jnt_qposadr)
        self.mj_jnt_qveladr = np.array(self.mj_jnt_qveladr)

        self.last_action = np.zeros(12, dtype=np.float64)
        self.rng = np.random.default_rng(0)
        self._new_target()
        self.reset()

    # ------- episode / target management -------
    def _new_target(self):
        self.dx = self.rng.uniform(*CMD_X)
        self.dy = self.rng.uniform(*CMD_Y)
        self.dyaw = self.rng.uniform(*CMD_YAW)
        self.target_time = self.rng.uniform(*CMD_T)
        self.step_count = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # place base at origin, standing height, identity yaw
        self.data.qpos[0:3] = [0.0, 0.0, 0.33]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, adr in enumerate(self.mj_jnt_qposadr):
            self.data.qpos[adr] = Q_DEFAULT[i]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # pin the world target from current base pose (base frame == world at reset)
        base_pos = self.data.qpos[0:3].copy()
        base_yaw = yaw_from_quat(self.data.qpos[3:7])
        cos_y, sin_y = math.cos(base_yaw), math.sin(base_yaw)
        self.target_pos_w = base_pos + np.array([
            cos_y * self.dx - sin_y * self.dy,
            sin_y * self.dx + cos_y * self.dy,
            0.0,
        ])
        self.target_yaw_w = wrap_to_pi(base_yaw + self.dyaw)
        self.last_action[:] = 0.0

    # ------- observation (51-D, exact Isaac order) -------
    def get_obs(self):
        q = self.data.qpos[self.mj_jnt_qposadr]              # 12, isaac order
        qd = self.data.qvel[self.mj_jnt_qveladr]             # 12
        quat = self.data.qpos[3:7].copy()                     # w,x,y,z
        lin_w = self.data.qvel[0:3].copy()
        ang_w = self.data.qvel[3:6].copy()

        base_lin_vel = quat_rotate_inverse(quat, lin_w)       # base frame
        base_ang_vel = quat_rotate_inverse(quat, ang_w)
        proj_grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        # --- 5D pose command (LocalPoseTimeCommand) ---
        base_pos = self.data.qpos[0:3].copy()
        base_yaw = yaw_from_quat(quat)
        cos_y, sin_y = math.cos(base_yaw), math.sin(base_yaw)
        err_w = self.target_pos_w - base_pos                  # world error
        local_dx = cos_y * err_w[0] + sin_y * err_w[1]        # yaw-frame error
        local_dy = -sin_y * err_w[0] + cos_y * err_w[1]
        local_dz = err_w[2]
        local_dyaw = wrap_to_pi(self.target_yaw_w - base_yaw)
        pose_cmd = np.array([local_dx, local_dy, local_dz, local_dyaw, self.target_time])

        # --- time_to_target ---
        t_elapsed = self.step_count * (SIM_DT * DECIMATION)
        t_rem = max(self.target_time - t_elapsed, 0.0) / self.target_time

        obs = np.concatenate([
            base_lin_vel,                 # 3
            base_ang_vel,                 # 3
            proj_grav,                    # 3
            pose_cmd,                     # 5
            q - Q_DEFAULT,                # 12  joint_pos_rel
            qd,                           # 12  joint_vel_rel (rel-to-zero default vel)
            self.last_action,             # 12
            np.array([t_rem]),            # 1
        ]).astype(np.float32)
        return torch.from_numpy(obs).unsqueeze(0)

    # ------- one control step (policy + 4 substeps, position servos) -------
    def step(self, hold=False):
        if hold:
            # DEBUG: ignore policy, just hold the default pose. If the robot
            # stands in this mode but falls with the policy, the bug is in the
            # observation vector, not the mechanical/actuator path.
            q_target = Q_DEFAULT.copy()
            self.last_action = np.zeros(12)
        else:
            with torch.inference_mode():
                action = self.policy(self.get_obs()).detach().numpy()[0].astype(np.float64)
            self.last_action = action.copy()
            q_target = Q_DEFAULT + ACTION_SCALE * action      # isaac order, target ANGLES

        # menagerie actuators are POSITION servos: write the target angle into
        # ctrl (NOT a torque). Map Isaac-order targets into MuJoCo order.
        for i, aid in enumerate(self.mj_act_id):
            self.data.ctrl[aid] = q_target[i]

        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

    # ------- live metrics for the on-screen overlay -------
    def metrics(self):
        base_xy = self.data.qpos[0:2].copy()
        base_yaw = yaw_from_quat(self.data.qpos[3:7])
        dist_err = float(np.linalg.norm(self.target_pos_w[:2] - base_xy))
        yaw_err = abs(wrap_to_pi(self.target_yaw_w - base_yaw))
        elapsed = self.step_count * (SIM_DT * DECIMATION)
        t_rem = max(self.target_time - elapsed, 0.0)
        return dict(target_time=self.target_time, elapsed=elapsed, t_rem=t_rem,
                    dist_err=dist_err, yaw_err=yaw_err)

    def check_episode_end(self):
        # episode end: reached target time budget + small hold, or fell
        base_z = self.data.qpos[2]
        elapsed = self.step_count * (SIM_DT * DECIMATION)
        if elapsed >= self.target_time + 1.0 or base_z < 0.15:
            dist = np.linalg.norm(self.target_pos_w[:2] - self.data.qpos[0:2])
            print(f"[EP] T={self.target_time:.2f}s  final dist err={dist*100:.1f} cm")
            self._new_target()
            self.reset()


def _add_frame(scn, pos, yaw, axis_len=0.25, radius=0.012):
    """Draw an RGB coordinate frame (x=red, y=green, z=blue) at pos with given
    yaw (rad) about z, appended to the viewer's user scene geoms."""
    c, s = math.cos(yaw), math.sin(yaw)
    # rotation matrix (yaw about z), column-major flattened for mjv_initGeom
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    axes = [
        (R @ np.array([1.0, 0, 0]), np.array([1.0, 0.2, 0.2, 1.0])),  # x red
        (R @ np.array([0, 1.0, 0]), np.array([0.2, 1.0, 0.2, 1.0])),  # y green
        (R @ np.array([0, 0, 1.0]), np.array([0.2, 0.4, 1.0, 1.0])),  # z blue
    ]
    for direction, rgba in axes:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        p1 = np.asarray(pos, dtype=np.float64)
        p2 = p1 + direction * axis_len
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.zeros(9),
            rgba.astype(np.float32))
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p1, p2)
        scn.ngeom += 1


def _draw_scene(runner, viewer):
    """Refresh user-scene geoms: robot-body frame + target frame."""
    scn = viewer.user_scn
    scn.ngeom = 0
    # robot body frame
    base_pos = runner.data.qpos[0:3].copy()
    base_yaw = yaw_from_quat(runner.data.qpos[3:7])
    _add_frame(scn, base_pos, base_yaw, axis_len=0.30)
    # target frame (slightly raised so it's visible on the ground)
    tgt = np.array([runner.target_pos_w[0], runner.target_pos_w[1], 0.05])
    _add_frame(scn, tgt, runner.target_yaw_w, axis_len=0.30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="exported/policy.pt (TorchScript)")
    p.add_argument("--xml", required=True, help="unitree_go1 scene.xml (menagerie)")
    p.add_argument("--hold", action="store_true",
                   help="DEBUG: ignore policy, hold default pose. Robot should stand.")
    p.add_argument("--print_obs", action="store_true",
                   help="DEBUG: print the first observation and exit.")
    p.add_argument("--topdown", action="store_true",
                   help="Use a fixed top-down (bird's-eye) camera. Ground stays "
                        "stationary; the robot moves across the frame.")
    p.add_argument("--follow", action="store_true",
                   help="With --topdown, track the robot (ground drifts). Off by "
                        "default so the ground is stationary.")
    args = p.parse_args()

    runner = Go1Runner(args.xml, args.policy)

    if args.print_obs:
        obs = runner.get_obs().numpy()[0]
        names = (["lin_vel"]*3 + ["ang_vel"]*3 + ["proj_grav"]*3 +
                 ["pose_cmd"]*5 + ["jpos_rel"]*12 + ["jvel"]*12 +
                 ["last_act"]*12 + ["t_rem"]*1)
        print("\n=== first observation (upright at spawn) ===")
        for i, (nm, v) in enumerate(zip(names, obs)):
            print(f"  [{i:2d}] {nm:10s} {v:+.4f}")
        print("\nSanity: proj_grav should be ~[0,0,-1]; lin_vel/ang_vel ~0; "
              "pose_cmd[0:2] ~ your (dx,dy).")
        return

    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        # ---- camera ----
        if args.topdown:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -89.0     # almost straight down
            viewer.cam.distance = 8.0
            # fixed look-at on the arena origin so the ground does NOT drift.
            # Targets are sampled within +/-1.5 m, so origin keeps everything
            # in frame. Widen --distance if targets leave the view.
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]

        while viewer.is_running():
            t0 = time.time()
            runner.step(hold=args.hold)

            # frames on robot body + target
            _draw_scene(runner, viewer)

            # follow the robot only if explicitly requested (ground drifts).
            # Default: fixed view, ground stays put, robot walks across frame.
            if args.topdown and args.follow:
                viewer.cam.lookat[0] = runner.data.qpos[0]
                viewer.cam.lookat[1] = runner.data.qpos[1]

            runner.check_episode_end()
            viewer.sync()

            # live metrics readout (target time / remaining / error).
            # The passive viewer has no stable in-window text API across MuJoCo
            # versions, so this prints to the console, updating in place.
            m = runner.metrics()
            if runner.step_count % 5 == 0:
                print(f"  T={m['target_time']:.2f}s  rem={m['t_rem']:.2f}s  "
                      f"err={m['dist_err']*100:5.1f}cm  yaw={math.degrees(m['yaw_err']):5.1f}deg   ",
                      end="\r", flush=True)

            dt = SIM_DT * DECIMATION - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
