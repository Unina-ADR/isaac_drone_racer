from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
import isaaclab.utils.math as utils

from dynamics import Allocation,Motor
from utils.logger import log
from controllers import BetaflightControllerParams, BetaflightController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv






class ControlAction(ActionTerm):
    r"""Body torque control action term.

    This action term applies a wrench to the drone body frame based on action commands

    """

    cfg: ControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._raw_actions_obs = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._last_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._allocation = Allocation(
            num_envs=self.num_envs,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._controllerparams = BetaflightControllerParams(num_envs=self.num_envs)
        self._controller = BetaflightController(params=self._controllerparams)

        #frame conversion utils
        self.flu_to_frd = torch.tensor(
            [[1, 0, 0],
             [0, -1, 0],
             [0, 0, -1]], device=self.device, dtype=torch.float32)
        
        

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property 
    def raw_actions_obs(self) -> torch.Tensor:
        return self._raw_actions_obs

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return True

    @property
    def elapsed_time(self) -> torch.Tensor:
        r"""Elapsed time since the last action was applied."""
        return self._elapsed_time
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):

        self._last_action.copy_(self._processed_actions)
        self._raw_actions_obs.copy_(self._raw_actions)
        self._raw_actions[:] = actions #.clamp_(-1.0, 1.0)
        clamped = self._raw_actions.clamp_(-1.0, 1.0)

        #NO BETAFLIGHT PROCESSING
        mapped = (clamped + 1.0) / 2.0
        omega_ref = self.cfg.omega_max * mapped
        
        omega_real = self._motor.compute(omega_ref)
        self._processed_actions = self._allocation.compute(omega_real)
        self._processed_actions[:,0] = self._processed_actions[:,0].clamp(0.0, self.cfg.max_thrust * self.cfg.thrust_saturation)


        # # #BETAFLIGHT PROCESSING
        # ang_vel_frd_b = utils.quat_rotate_inverse(self._robot.data.root_quat_w, self._robot.data.root_ang_vel_w)
        # ang_vel_frd = torch.bmm(
        #     ang_vel_frd_b.unsqueeze(1),
        #     self.flu_to_frd.unsqueeze(0).repeat(self.num_envs, 1, 1)
        # ).squeeze(1)
        # #ang_vel_frd = self.flu_to_frd * utils.quat_rotate_inverse(drone_quat, ang_vel_flu)
        # ang_vel_des, omega_ref_pwm = self._controller.compute(ang_vel_frd)


        # omega_ref = omega_ref_pwm * self.cfg.omega_max
        # omega_real = self._motor.compute(omega_ref)
        
        # self._processed_actions= self._allocation.compute(omega_real)  # (num_envs,4): [T, Mx, My, Mz] in FRD

        #log(self._env, ["wx_des", "wy_des", "wz_des"], ang_vel_des)
        #log(self._env, ["A", "E", "T", "R"], self._raw_actions)
        log(self._env, ["thrust", "moment_x", "moment_y", "moment_z"], self._processed_actions)
        log(self._env, ["w1", "w2", "w3", "w4"], omega_real)

    def apply_actions(self):
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._motor.reset(env_ids)
        self._robot.reset(env_ids)
        self._controller.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "force_visualizer"):
                # usa il config freccia già fornito
                cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Actions/force")
                # opzionale: modifiche al prototipo se compatibile
                try:
                    cfg.markers["arrow"].scale = (1.0, 0.1, 0.1)
                except Exception:
                    pass
                self.force_visualizer = VisualizationMarkers(cfg)
            self.force_visualizer.set_visibility(True)
        else:
            if hasattr(self, "force_visualizer"):
                self.force_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        # posizione del corpo (in world)
        pos = self._robot.data.root_pos_w  # (num_envs, 3)

        # forza applicata in body frame
        force_b = self._thrust[:, 0, :]  # (num_envs,3)

        # trasformala in world
        force_w = utils.quat_rotate(self._robot.data.root_quat_w, force_b)

        # magnitudine e direzione
        lengths = torch.norm(force_w, dim=1)  # (num_envs,)
        eps = 1e-8
        dirs = force_w / lengths.unsqueeze(1).clamp(min=eps)

        # costruisci quaternioni che ruotano +X verso dirs
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        dots = (x_axis * dirs).sum(dim=1).clamp(-1.0, 1.0)
        angles = torch.acos(dots)
        axes = torch.cross(x_axis, dirs)
        axes_norm = axes / axes.norm(dim=1, keepdim=True).clamp(min=eps)
        quats = utils.quat_from_angle_axis(angles, axes_norm)
        small_mask = lengths < 1e-6
        if small_mask.any():
            quats[small_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # scale per le frecce: componi (lunghezza, spessore_y, spessore_z)
        length_scale = 0.1
        scales = torch.stack([
            lengths * length_scale,
            torch.ones_like(lengths) * 0.1,
            torch.ones_like(lengths) * 0.1
        ], dim=1)

        # converti su CPU / numpy prima di visualizzare
        pos_np = pos.cpu().numpy()
        quats_np = quats.cpu().numpy()
        scales_np = scales.cpu().numpy()

        self.force_visualizer.visualize(
            translations=pos_np,
            orientations=quats_np,
            scales=scales_np
        )


@configclass
class ControlActionCfg(ActionTermCfg):
    """
    See :class:`ControlAction` for more details.
    """

    class_type: type[ActionTerm] = ControlAction
    """ Class of the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    arm_length: float = 0.1825
    """Length of the arms of the drone in meters."""
    #drag_coef: float = 8.97e-9 #CALCOLATO COL SENSORE ATI
    drag_coef: float = 7.6909678e-9
    """Drag torque coefficient."""
    #thrust_coef: float = 6.888e-7 #CALCOLATO COL SENSORE ATI
    thrust_coef: float = 1.52117969e-5
    """Thrust coefficient.
    Calculated with 5145 rad/s max angular velociy, thrust to weight: 7.69, mass: 0.8702 kg and gravity: 9.81 m/s^2.
    thrust_coef = (7.69 * 0.8702 * 9.81) / (4 * 5541**2) = 5.3453e-7."""
    #omega_max: float = 5541.0 # Full battery
    omega_max: float = 1885.0
    """Maximum angular velocity of the drone motors in rad/s.
    Calculated with 2100KV motor, with 6S LiPo battery with 3.8V per cell.
    2100 * 6 * 4.2 = 47,780 RPM ~= 5541 rad/s."""
    max_thrust: float = omega_max**2 * thrust_coef * 4
    """Maximum collective thrust in Newtons."""
    thrust_saturation: float = 1.0
    """Saturation limit for collective thrust as a fraction of maximum thrust."""
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    """Time constants for each motor."""
    init: list[float] = (100.0, 100.0, 100.0, 100.0)
    """Initial angular velocities for each motor in rad/s."""
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    use_motor_model: bool = True
    """Flag to determine if motor delay is bypassed."""
