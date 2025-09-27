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
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's is too far away from the target position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute distance
    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance


def out_of_bounds(
    env: ManagerBasedRLEnv,
    bounds: tuple[float, float], # (y_max, z_max)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's position is out of the specified bounds."""

    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the asset's position
    pos_y = asset.data.root_pos_w[:, 1] # Get the y position
    pos_z = asset.data.root_pos_w[:, 2] # Get the z position

    # Check if the position is out of bounds
    #out_of_bounds_x = (pos[:, 0] < -bounds[0]) | (pos[:, 0] > bounds[0])
    out_of_bounds_y = (pos_y[:] < -bounds[0]) | (pos_y[:] > bounds[0])
    out_of_bounds_z = (pos_z[:] < 0.1) | (pos_z[:] > bounds[1])

    return out_of_bounds_y | out_of_bounds_z

def dynamic_limits_exceeded(env: ManagerBasedRLEnv, linvel_max: float = 20.0, angvel_max: float = 11.69, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Terminate when the asset's linear or angular velocity exceeds specified limits."""

    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the asset's linear and angular velocities
    lin_vel = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1)
    ang_vel = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)

    # Check if the velocities exceed the limits
    lin_vel_exceeded = lin_vel > linvel_max
    ang_vel_exceeded = ang_vel > angvel_max

    return lin_vel_exceeded | ang_vel_exceeded


# def time_out(
#     env: ManagerBasedRLEnv,
#     command_name: str = "target",
#     timeout_s: float = 5.0,
# ) -> torch.Tensor:
#     """
#     Returns True for envs where the target_pos command has not been updated for timeout_s seconds.
#     Assumes env has .sim_time and command manager/term tracks last update time per env.
#     """
#     # Get the command term
#     command_term = env.command_manager.get_term(command_name)
#     elapsed_time = env.action_manager.get_term("control_action").elapsed_time
#     # You must ensure last_update_time is tracked in your command term (shape: [num_envs])
#     # For example, update command_term.last_update_time[env_ids] = env.sim_time when command is updated

#     # Compute time since last update for each env
#     time_since_update = env.sim_time - command_term.last_update_time  # [num_envs]
#     # Return mask: True if timed out
#     return time_since_update > timeout_s