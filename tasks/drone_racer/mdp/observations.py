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

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    log(env, ["vx", "vy", "vz"], lin_vel)
    #return lin_vel.clamp_(min=-1.0, max=1.0)  # Clamp to avoid NaN issues in training
    return lin_vel


def root_lin_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_w
    log(env, ["vx", "vy", "vz"], lin_vel)
    #return scaled_vel.clamp_(min=-1.0, max=1.0)  # Clamp to avoid NaN issues in training
    return lin_vel

def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ang_vel_max: float = 11.0) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    scaled_ang_vel = ang_vel / ang_vel_max  # Scale to [-1, 1]
    log(env, ["wx", "wy", "wz"], ang_vel)
    #return scaled_ang_vel.clamp_(min=-1.0, max=1.0)
    return ang_vel

def root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    log(env, ["qw", "qx", "qy", "qz"], quat)
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return flat_rotmat


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pos_max : float = 30.0) -> torch.Tensor:
    """Asset root position in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    scaled_position = position / pos_max  # Scale to [-1, 1]
    #return position.clamp_(min=-1.0, max=1.0)
    return position

def root_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root position in the gate frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    drone_pose_w = asset.data.root_state_w[:, :7]  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    drone_pos_w = drone_pose_w[:, :3]
    drone_quat_w = drone_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = drone_pos_w - gate_pos_w
    drone_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    drone_quat_g = math_utils.quat_mul(gate_quat_w_inv, drone_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([drone_pos_g, drone_quat_g], dim=-1)

    #return position.clamp_(min=-1.0, max=1.0)
    return position


def action_obs(env:ManagerBasedRLEnv) -> torch.Tensor:
    """Last raw action taken by the agent."""
    action_term = env.action_manager.get_term("control_action")
    action = action_term.raw_actions_obs  # [num_envs, 4]
    #action = action_term.processed_actions  # [num_envs, 4]
    return action

# def next_gate_pose_g(
#     env: ManagerBasedRLEnv,
#     command_name: str,
# ) -> torch.Tensor:
#     """Asset root position in the gate frame."""
#     gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
#     #next_gate_pose_w = env.command_manager.get_term(command_name).next_gate  # (num_envs, 7)

#     # Extract positions and quaternions
#     gate_pos_w = gate_pose_w[:, :3]
#     gate_quat_w = gate_pose_w[:, 3:7]
#     #next_gate_pos_w = next_gate_pose_w[:, :3]
#     #next_gate_quat_w = next_gate_pose_w[:, 3:7]

#     # Compute drone pose in gate frame
#     # Inverse gate quaternion
#     gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

#     # Position of drone in gate frame
#     rel_pos = next_gate_pos_w - gate_pos_w
#     next_gate_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

#     # Orientation of drone in gate frame
#     #next_gate_quat_g = math_utils.quat_mul(gate_quat_w_inv, next_gate_quat_w)

#     # Concatenate position and quaternion
#     #position = torch.cat([next_gate_pos_g, next_gate_quat_g], dim=-1)

#     return next_gate_pos_g.clamp_(min=-1.0, max=1.0)


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    pos_max: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) ->  torch.Tensor:
    """Position of target in body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
        next_target_pos = env.command_manager.get_term(command_name).next_command[:, :3]
        next_target_pos_tensor = next_target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )
        next_target_pos_tensor = (torch.tensor(next_target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor)

    pos_b = pos_b / pos_max  # Scale to [-1, 1]
    next_pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, next_target_pos_tensor)
    next_pos_b = next_pos_b / pos_max  # Scale to [-1, 1]
    return torch.cat([pos_b, next_pos_b], dim=-1)  # [num_envs, 6]
    #return pos_b.clamp_(min=-1.0, max=1.0)
    #return pos_b


def waypoint_obs(
    env: ManagerBasedRLEnv,
    command_name,  # Pass the GateTargetingCommand instance here
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    l_max: float = 10.0,
) -> torch.Tensor:
    """34-dimensional waypoint observation using current and next gate indices from command_term."""
    asset: RigidObject = env.scene[asset_cfg.name]
    track = env.scene["track"]

    gate_inner_size = 1.5
    drone_pos = asset.data.root_pos_w

    gate_positions = track.data.object_com_pos_w  # [num_envs, num_gates, 3]
    gate_orientations = track.data.object_quat_w  # [num_envs, num_gates, 4]

    # Get indices from the command term
    #current_gate_idx = env.command_manager.get_term(command_name).command  # [num_envs]
    next_gate_idx = env.command_manager.get_term(command_name).next_gate_idx        # [num_envs]
    gate_after_idx = env.command_manager.get_term(command_name).gate_after_idx  # [num_envs]
    # Gather positions and orientations for current and next gates
    #current_gate_pos = gate_positions[torch.arange(env.num_envs), current_gate_idx]  # [num_envs, 3]
    #current_gate_quat = gate_orientations[torch.arange(env.num_envs), current_gate_idx]  # [num_envs, 4]
    next_gate_pos = gate_positions[torch.arange(env.num_envs), next_gate_idx]  # [num_envs, 3]
    next_gate_quat = gate_orientations[torch.arange(env.num_envs), next_gate_idx]  # [num_envs, 4]

    gate_after_pos = gate_positions[torch.arange(env.num_envs), gate_after_idx]  # [num_envs, 3]
    gate_after_quat = gate_orientations[torch.arange(env.num_envs), gate_after_idx]

    # waypoint_obs = waypoint_obs_jit(
    #     drone_pos=drone_pos,
    #     next_gate_pos=current_gate_pos,
    #     next_gate_quat=current_gate_quat,
    #     gate_inner_size=gate_inner_size,
    #     l_max=l_max
    # )

    next_waypoint_obs = waypoint_obs_jit(
        drone_pos=drone_pos,
        next_gate_pos=next_gate_pos,
        next_gate_quat=next_gate_quat,
        gate_inner_size=gate_inner_size,
        l_max=l_max
    )

    gate_after_obs = waypoint_obs_jit(
        drone_pos=drone_pos,
        next_gate_pos=gate_after_pos,  # Using the same next gate position for now
        next_gate_quat=gate_after_quat,  # Using the same next gate quaternion for now
        gate_inner_size=gate_inner_size,
        l_max=l_max
    )
    #return torch.cat([waypoint_obs, next_waypoint_obs], dim=1)  # [num_envs, 34]
    return torch.cat([next_waypoint_obs, gate_after_obs], dim=1)  # [num_envs, 34]

@torch.jit.script
def waypoint_obs_jit(
    drone_pos: torch.Tensor,
    next_gate_pos: torch.Tensor,
    next_gate_quat: torch.Tensor,
    gate_inner_size: float = 1.5,
    l_max: float = 6.0
) -> torch.Tensor:
    num_envs = drone_pos.shape[0]
    # [4, 3] corners in gate frame
    relative_gate_corners = torch.tensor(
        [
            [0.0, -gate_inner_size/2, -gate_inner_size/2],
            [0.0, -gate_inner_size/2,  gate_inner_size/2],
            [0.0, gate_inner_size/2,   gate_inner_size/2],
            [0.0, gate_inner_size/2,  -gate_inner_size/2],
        ],
        dtype=drone_pos.dtype,
        device=drone_pos.device,
    ).unsqueeze(0).expand(num_envs, -1, -1)  # [num_envs, 4, 3]

    # Flatten for batch processing
    relative_gate_corners_flat = relative_gate_corners.reshape(-1, 3)  # [num_envs*4, 3]
    next_gate_quat_expand = next_gate_quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)  # [num_envs*4, 4]
    gate_corners_w_flat = math_utils.quat_apply(next_gate_quat_expand, relative_gate_corners_flat)  # [num_envs*4, 3]
    gate_corners_w = gate_corners_w_flat.reshape(num_envs, 4, 3) + next_gate_pos.unsqueeze(1)  # [num_envs, 4, 3]

    noise_std = 0.05 #metri

    gate_corners_w += torch.randn_like(gate_corners_w) * noise_std

    # Gate x axis in world frame
    gate_x_axis = math_utils.quat_apply(
        next_gate_quat,
        torch.tensor([1.0, 0.0, 0.0], dtype=drone_pos.dtype, device=drone_pos.device).expand(next_gate_quat.shape[0], 3)
    )
    gate_to_drone_vector = drone_pos - next_gate_pos

    # --- Compute cosine similarity ---
    dot_product = torch.sum(gate_x_axis * gate_to_drone_vector, dim=1)
    norm1 = torch.norm(gate_x_axis, dim=1)
    norm2 = torch.norm(gate_to_drone_vector, dim=1)
    s_c = (dot_product / (norm1 * norm2)).unsqueeze(1)  # [num_envs, 1]
    s_c=torch.nan_to_num(s_c, nan=1.0, posinf=1.0, neginf=-1.0)  # Handle NaN and Inf values
    # --- Compute corner vectors and norms ---

    corner_vectors = gate_corners_w - drone_pos.unsqueeze(1)  # [num_envs, 4, 3]
    #corner_vectors = torch.normalize(corner_vectors, dim=-1)  # Normalize to get unit vectors
    corner_vectors = corner_vectors / torch.norm(corner_vectors, dim=-1, keepdim=True)  # Normalize to get unit vectors
    vector_norms = torch.norm(corner_vectors, dim=-1)  # [num_envs, 4]
    corner_vectors_flat = corner_vectors.reshape(num_envs, -1)  # [num_envs, 12]
    vector_norms_scaled = torch.min(vector_norms/l_max, torch.ones_like(vector_norms))  # Scale norms to [0, 1]

    # --- Concatenate ---
    waypoint_obs = torch.cat([s_c, vector_norms_scaled, corner_vectors_flat], dim=1)  # [num_envs, 17]
    return waypoint_obs