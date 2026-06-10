import torch
from isaaclab.managers.command_manager import CommandTerm, CommandTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_yaw, yaw_quat, euler_xyz_from_quat, quat_apply_inverse


# =========================================================
# 1. DEFINE THE COMMAND EXECUTOR FIRST
# =========================================================
class LocalPoseCommand(CommandTerm):
    """Custom command generator that secretly holds a global pin, but outputs the dynamic local error."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # The public command the neural net and rewards see: [dx_local, dy_local, dz_local, dyaw_local]
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        
        # The HIDDEN global pin coordinates
        self._global_target_pose = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The current command."""
        return self._command

    def _update_metrics(self): pass

    def _update_command(self):
        """🔥 RECALCULATES THE LOCAL ERROR EVERY SINGLE SIMULATION STEP 🔥"""
        asset = self._env.scene["robot"]
        yaw_q = yaw_quat(asset.data.root_quat_w)

        # Calculate how far the robot currently is from the hidden global pin
        pos_error_w = self._global_target_pose[:, :3] - asset.data.root_pos_w
        local_pos_error = quat_apply_inverse(yaw_q, pos_error_w)

        # Calculate the current yaw error
        _, _, robot_yaw = euler_xyz_from_quat(yaw_q)
        local_yaw_error = self._global_target_pose[:, 3] - robot_yaw
        local_yaw_error = torch.atan2(torch.sin(local_yaw_error), torch.cos(local_yaw_error))

        # Update the public command buffer with the shrinking distances!
        self._command[:, 0:3] = local_pos_error
        self._command[:, 3] = local_yaw_error

    def _resample_command(self, env_ids: torch.Tensor):
        r = self.cfg.ranges
        num_resets = len(env_ids)
        
        # 1. Sample Local Offsets
        dx = torch.rand(num_resets, device=self.device) * (r.pos_x[1] - r.pos_x[0]) + r.pos_x[0]
        dy = torch.rand(num_resets, device=self.device) * (r.pos_y[1] - r.pos_y[0]) + r.pos_y[0]
        dz = torch.rand(num_resets, device=self.device) * (r.pos_z[1] - r.pos_z[0]) + r.pos_z[0]
        dyaw = torch.rand(num_resets, device=self.device) * (r.heading[1] - r.heading[0]) + r.heading[0]
        
        # 2. Get Robot's current state to "drop the pin"
        asset = self._env.scene["robot"]
        current_pos = asset.data.root_pos_w[env_ids]
        current_quat = asset.data.root_quat_w[env_ids]
        current_yaw_q = yaw_quat(current_quat)
        _, _, current_yaw_rad = euler_xyz_from_quat(current_quat)
        
        # 3. Transform to Global Targets
        local_pos_offset = torch.stack([dx, dy, dz], dim=1)
        global_target_pos = current_pos + quat_apply_yaw(current_yaw_q, local_pos_offset)
        
        global_target_yaw = current_yaw_rad + dyaw
        global_target_yaw = torch.atan2(torch.sin(global_target_yaw), torch.cos(global_target_yaw))
        
        # 4. Save to the HIDDEN Global Buffer
        self._global_target_pose[env_ids, 0:3] = global_target_pos
        self._global_target_pose[env_ids, 3] = global_target_yaw

        # 5. Immediately update the public command so step 0 is correct
        self._update_command()

# =========================================================
# 2. DEFINE THE CONFIG SECOND
# =========================================================
@configclass
class LocalPoseCommandCfg(CommandTermCfg):
    class_type: type = LocalPoseCommand 
    
    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (-1.5, 1.5)
        pos_y: tuple[float, float] = (-1.5, 1.5)
        pos_z: tuple[float, float] = (0.0, 0.0)
        heading: tuple[float, float] = (-3.14159, 3.14159) # Raw Yaw

    ranges: Ranges = Ranges()
    resampling_time_range: tuple[float, float] = (10000.0, 10000.0)