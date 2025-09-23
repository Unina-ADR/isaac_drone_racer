# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute sum of squared errors
    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    progress = prev_distance - current_distance

    return progress


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward for passing a gate."""
    missed = (-1.0) * env.command_manager.get_term(command_name).gate_missed
    passed = (1.0) * env.command_manager.get_term(command_name).gate_passed
    return missed + passed
    #return passed

def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for looking at the next gate."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = next_gate_pos - drone_pos
    vec_to_gate = math_utils.normalize(vec_to_gate)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    # The camera is rotated 50 degrees around the y-axis
    
    cam_roll = torch.tensor([0.0], device=asset.device).expand(env.num_envs, 1)
    cam_pitch = torch.tensor([-0.87266], device=asset.device).expand(env.num_envs, 1)
    cam_yaw = torch.tensor([0.0], device=asset.device).expand(env.num_envs, 1)
    cam_quat = math_utils.quat_from_euler_xyz(cam_roll, cam_pitch, cam_yaw)
    drone_x_axis = math_utils.quat_apply(drone_att, x_axis)
    drone_x_axis = math_utils.normalize(drone_x_axis)
    cam_x_axis = math_utils.quat_apply(cam_quat, drone_x_axis)
    cam_x_axis = math_utils.normalize(cam_x_axis)
    

    #dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    dot = (cam_x_axis * vec_to_gate).sum(dim=1)
    cosine= dot/(torch.norm(cam_x_axis, dim=1)* torch.norm(vec_to_gate, dim=1))
    cosine = torch.clamp(cosine, -1.0, 1.0)  # Ensure cosine is within [-1, 1]
    #sign = torch.sign(dot)
    angle = torch.acos(cosine)
    #angle.nan_to_num_(nan=0.0)  # Replace NaN with 0.0
    expangle = torch.pow(angle, 4)
    #expangle = torch.pow(angle, 2)
    return torch.exp(expangle * std)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)

def yaw_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel= asset.data.root_ang_vel_b
    # Only consider the yaw component of the angular velocity
    ang_vel_yaw = ang_vel[:, 2]  # Assuming the yaw is the third component
    # Square the yaw component and sum it across the batch
    return torch.abs(ang_vel_yaw)

def action_reward(
    env: ManagerBasedRLEnv,
    #action: ControlAction | None = None,
    weight_omega: float = 0.01,
    weight_rate: float = 0.01,
    #asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for the action taken by the agent."""
    # extract the used quantities (to enable type-hinting)
    #asset: RigidObject = env.scene[asset_cfg.name]
    
    #action = asset.action_manager.get_term("action").processed_actions
    # Compute the L2 norm of the action
    #action= env.action_manager.get_term("action").process_actions(action)
    #last_action = asset.data.last_action
    action_term = env.action_manager.get_term("control_action")
    action = action_term._processed_actions
    last_action = action_term._last_action
    action_omega = action[:, 1:]
    action_rate = torch.square(torch.norm(action - last_action, dim=1))

    omega_rew = weight_omega*torch.norm(action_omega, dim=1)
    return weight_omega * omega_rew  + weight_rate * action_rate



def linear_vel_forward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    #command_name: str | None = None,
    #weight1: float = 0.0,
    #weight2: float = 0.0,
) -> torch.Tensor:
    """Reward for the linear velocity of the drone."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # get drone velocity
    drone_linear_vel = asset.data.root_lin_vel_b
    v_x = drone_linear_vel[:, 0]

    return torch.square(torch.min(v_x, torch.zeros_like(v_x)))

def linear_vel_side(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    drone_linear_vel = asset.data.root_lin_vel_b
    v_y = drone_linear_vel[:, 1]
    return torch.square(v_y)


def lin_vel_to_next_gate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward for the linear velocity of the drone towards the next gate."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # get drone velocity
    drone_linear_vel = asset.data.root_lin_vel_b
    vec_to_next_gate = env.command_manager.get_term(command_name).point_to_next_gate

    # Compute the dot product between the drone's linear velocity and the vector to the next gate
    dot_product = torch.sum(drone_linear_vel * vec_to_next_gate, dim=1)

    return dot_product

def time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for the time elapsed since the last action was applied."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # get the elapsed time
    elapsed_time = env.action_manager.get_term("control_action").elapsed_time
    return elapsed_time.squeeze(-1)  # If elapsed_time is [num_envs, 1], squeeze it to [num_envs]

def pitch_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for the pitch angle of the drone."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # get drone orientation
    drone_att = asset.data.root_quat_w
    (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(drone_att) # Pitch is the second component
    max_pitch =  1.22 # Maximum pitch angle in radians (70 degrees)
    return torch.abs(pitch) > max_pitch # This will return 1 if the pitch is greater than max_pitch, otherwise 0


def guidance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    guidance_x_thresh: float = 3.0, 
    guidance_tol: float = 0.2,
    k_rejection: float = 2.0,
) -> torch.Tensor:
    """Reward for the guidance of the drone."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    drone_pos = asset.data.root_pos_w

    gate_to_drone = drone_pos - next_gate_pos
    gate_quat = env.command_manager.get_term(command_name).command[:, 3:7]

    gate_quat_inverse = math_utils.quat_inv(gate_quat)
    drone_pos_gate_frame= math_utils.quat_apply(gate_quat_inverse, gate_to_drone)

    guidance_reward = compute_guidance_reward(
        drone_pos_gate_frame,
        gate_height=1.5*torch.ones(env.num_envs, device=asset.device),
        gate_width=1.5*torch.ones(env.num_envs, device=asset.device),
        guidance_x_thresh=guidance_x_thresh,
        guidance_tol=guidance_tol,
        k_rejection=k_rejection,
    )
    return guidance_reward

    

@torch.jit.script
def compute_guidance_reward(
    drone_pos_gate_frame: torch.Tensor,
    gate_height: torch.Tensor,
    gate_width: torch.Tensor,
    guidance_x_thresh: float = 3.0,
    guidance_tol: float = 0.2,
    k_rejection: float = 2.0,
) -> torch.Tensor:
    x, y, z = (drone_pos_gate_frame[:, i] for i in range(3))

    layer_x = -torch.sgn(x) / guidance_x_thresh * x +1
    layer_x.clamp_(min=0.0)
    guidance_x = -(layer_x**2)

    tol = torch.where(x>0, 0.5, guidance_tol)

    yz_scale = (
        (1 - guidance_x) * tol * ((z**2 + y**2) / ((z / gate_height) ** 2 + (y / gate_width) ** 2)) ** 0.5
    )
    yz_scale.nan_to_num_(nan=1.0)  # caused by z**2 + y**2 == 0
    guidance_yz = torch.where(
        x > 0,
        k_rejection * torch.exp(-0.5 * (y**2 + z**2) / yz_scale),
        (1 - torch.exp(-0.5 * (y**2 + z**2) / yz_scale)),
    )

    guidance = guidance_x + guidance_yz

    return guidance