# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

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
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    @property
    def elapsed_time(self) -> torch.Tensor:
        r"""Elapsed time since the last action was applied."""
        return self._elapsed_time
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._last_action.copy_(self._processed_actions)
        # self._raw_actions[:] = actions
        # clamped = self._raw_actions.clamp_(-1.0, 1.0)
        # self._controller.set_command(clamped)
        # #ang_vel is in FLU frame, we need to convert it to FRD frame
        # ang_vel_flu = self._robot.data.root_ang_vel_b
        # ang_vel_frd = torch.bmm(
        #     ang_vel_flu.unsqueeze(1),
        #     self.flu_to_frd.unsqueeze(0).repeat(self.num_envs, 1, 1)
        # ).squeeze(1)
        # ang_vel_des, omega_ref_pwm = (
        #     self._controller.compute(ang_vel_frd)
        # )

        # omega_ref = omega_ref_pwm * self.cfg.omega_max
        # omega_real = self._motor.compute(omega_ref)
        # #everything is FRD to this point; we need to convert to FLU
        # if self.cfg.use_motor_model == True:
        #    omega_real = self._motor.compute(omega_ref)
        # else:
        #    omega_real = omega_ref

        # thrust_torque_frd = self._controller.get_thrust_and_torque_command(self.cfg.omega_max, self.cfg.thrust_coef, self.cfg.drag_coef, omega_ref_pwm)
        # thrust_flu = thrust_torque_frd[:, 0]
        # moment_frd = thrust_torque_frd[:, 1:]
        # moment_flu = torch.bmm(
        #     moment_frd.unsqueeze(1),
        #     self.flu_to_frd.unsqueeze(0).repeat(self.num_envs, 1, 1)
        # ).squeeze(1)
        # self._processed_actions = torch.cat([thrust_flu.unsqueeze(1), moment_flu], dim=1)

        self._raw_actions[:] = actions
        clamped = self._raw_actions.clamp_(-1.0, 1.0)
        mapped = (clamped + 1.0) / 2.0
        omega_ref = self.cfg.omega_max * mapped
        
        omega_real = self._motor.compute(omega_ref)
        self._processed_actions = self._allocation.compute(omega_real)


        #log(self._env, ["wx_des", "wy_des", "wz_des"], ang_vel_des)
        log(self._env, ["A", "E", "T", "R"], self._raw_actions)
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
    drag_coef: float = 1.5e-9
    """Drag torque coefficient."""
    thrust_coef: float = 5.3453e-7
    """Thrust coefficient.
    Calculated with 5145 rad/s max angular velociy, thrust to weight: 7.69, mass: 0.8702 kg and gravity: 9.81 m/s^2.
    thrust_coef = (7.69 * 0.8702 * 9.81) / (4 * 5541**2) = 5.3453e-7."""
    omega_max: float = 5541.0
    """Maximum angular velocity of the drone motors in rad/s.
    Calculated with 2100KV motor, with 6S LiPo battery with 4.2V per cell.
    2100 * 6 * 4.2 = 52,920 RPM ~= 5541 rad/s."""
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    """Time constants for each motor."""
    init: list[float] = (2572.5, 2572.5, 2572.5, 2572.5)
    """Initial angular velocities for each motor in rad/s."""
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""
