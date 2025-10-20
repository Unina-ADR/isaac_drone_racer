from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def root_lin_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_w
    log(env, ["vx", "vy", "vz"], lin_vel)
    #return scaled_vel.clamp_(min=-1.0, max=1.0)  # Clamp to avoid NaN issues in training
    return lin_vel

def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    log(env, ["vx", "vy", "vz"], lin_vel)
    #return lin_vel.clamp_(min=-1.0, max=1.0)  # Clamp to avoid NaN issues in training
    return lin_vel

def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
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

def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pos_max : float = 30.0) -> torch.Tensor:
    """Asset root position in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    #return position.clamp_(min=-1.0, max=1.0)
    return position

def action_obs(env:ManagerBasedRLEnv) -> torch.Tensor:
    """Last raw action taken by the agent."""
    action_term = env.action_manager.get_term("control_action")
    action = action_term.raw_actions_obs  # [num_envs, 4]
    #action = action_term.processed_actions  # [num_envs, 4]
    return action

def dist_from_goal_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-env distance to goal (default origin). Returns shape (num_envs, 1)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w  # (N, 3)
    if goal is None:
        goal_pos = torch.zeros_like(pos)
    else:
        # broadcast / reshape goal to (N, 3)
        if goal.ndim == 1:
            goal_pos = goal.unsqueeze(0).expand_as(pos)
        else:
            goal_pos = goal.expand_as(pos)
    dist = torch.linalg.norm(pos - goal_pos, dim=1, keepdim=True)  # (N, 1)
    log(env, ["dist_goal"], dist)
    return dist


def goal_pos_in_body_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return the goal point position expressed in the robot's body frame.

    - Output shape: (N, 3)  (N = num_envs)
    - Uses isaaclab.utils.math.subtract_frame_transforms under the hood.
      This function computes T_12 = T_01^{-1} * T_02, so passing
      robot pose as frame 01 and goal pose as frame 02 yields the
      goal pose *expressed in the robot frame* (translation = what we want). :contentReference[oaicite:1]{index=1}
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w        # (N, 3)
    quat_w = asset.data.root_quat_w      # (N, 4) - IsaacLab uses (w, x, y, z) convention. :contentReference[oaicite:2]{index=2}

    # Prepare goal_pos: allow goal to be None, (3,), (1,3), or (N,3)
    if goal is None:
        goal_pos_w = torch.zeros_like(pos_w)
    else:
        goal = torch.as_tensor(goal, device=pos_w.device, dtype=pos_w.dtype)
        if goal.ndim == 1 and goal.numel() == 3:
            # single goal for all envs -> expand
            goal_pos_w = goal.unsqueeze(0).expand_as(pos_w)
        elif goal.ndim == 2 and goal.shape == pos_w.shape:
            # per-env goals already
            goal_pos_w = goal
        else:
            # try a safe broadcast and error clearly if impossible
            try:
                goal_pos_w = goal.expand_as(pos_w)
            except Exception as e:
                raise ValueError(
                    f"Cannot broadcast `goal` with shape {tuple(goal.shape)} to required (N,3)"
                ) from e

    # If a goal orientation is not provided, use identity quaternion for the goal frame.
    # Keep dtype/device consistent with asset quaternions.
    N = pos_w.shape[0]
    goal_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat_w.device, dtype=quat_w.dtype)
    goal_quat_w = goal_quat_w.unsqueeze(0).expand(N, 4)  # (N,4)

    # Compute transform of goal frame relative to robot frame:
    # t_bg is translation of goal frame expressed in robot (body) frame -> shape (N,3)
    t_bg, q_bg = math_utils.subtract_frame_transforms(pos_w, quat_w, goal_pos_w, goal_quat_w)

    # t_bg is exactly the goal point position in the robot's body frame
    # Return it with shape (N,3)
    return t_bg
