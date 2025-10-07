from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.managers import ManagerBasedRLEnv 


def r_dist(env: ManagerBasedRLEnv, 
           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
           command: str = "goal",
           ) -> torch.Tensor:
    """Reward based on distance to target."""

    goal = env.command_manager.get_term(command).command
    asset: RidigObject = env.scene[asset_cfg.name]

    error_norm = torch.linalg.norm((goal - asset.data.root_pos_w),dim = 1)

    return (1.0- torch.tanh(error_norm))
    #return error_norm

def r_upright(env: ManagerBasedRLEnv,
           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
           ) -> torch.Tensor:
    """Reward based on uprightness of the lander."""

    asset: RidigObject = env.scene[asset_cfg.name]

    # Get the up vector of the lander in world frame
    rotmat = math_utils.matrix_from_quat(asset.data.root_quat_w) # Assuming Z-axis is the 'up' direction

    up_vector = rotmat[:, :, 2]

    # Compute the dot product with the world up vector (0, 0, 1)
    world_up = torch.tensor([0, 0, 1], device=asset.device, dtype=up_vector.dtype)
    dot_product = torch.einsum('ij,j->i', up_vector, world_up)

    # Reward is higher when the lander is more upright (dot product closer to 1)
    r_up= (dot_product + 1) / 2  # Normalize to [0, 1]

    return r_up

def r_vsoft(env:ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            command: str = "goal"
            ) -> torch.Tensor:
    """Reward encouraging slowing down while landing"""
    asset: RidigObject = env.scene[asset_cfg.name]

    vz = asset.data.root_lin_vel_w[:,2]
    height = asset.data.root_pos_w[:,2]

    goal = env.command_manager.get_term(command).command
    dist= torch.linalg.norm((goal - asset.data.root_pos_w),dim = 1)
    near = torch.exp(-((dist)**2)) * torch.exp(-((height - 0.1) ** 2) / (0.5**2))
    r_vsoft = (1.0 - torch.tanh(vz)) * near 
    return r_vsoft

def r_hspeed(env:ManagerBasedRLEnv,
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
             ) -> torch.Tensor:
    """"Reward penalizes horizontal speeds"""
    
    asset: RidigObject = env.scene[asset_cfg.name]

    hspeed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)

    return hspeed

# def r_effort(env:ManagerBasedRLEnv,
#              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#              ) -> torch.Tensor:
#     """Reward penalizes control effort"""
    
#     asset: RidigObject = env.scene[asset_cfg.name]

#     effort = torch.linalg.norm(asset.data.joint_effort, dim=1)

#     return effort

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

def r_penalize_upward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_squared: bool = True,
):
    """
    Returns upward-velocity penalty (positive) with shape (N,1)
    """
    asset = env.scene[asset_cfg.name]
    # root_lin_vel_w is (N, 3)
    vz = asset.data.root_lin_vel_w[:, 2]        # (N,)
    upward = torch.clamp(vz, min=0.0)            # (N,)

    if use_squared:
        pen = upward * upward                     # (N,)
    else:
        pen = upward                              # (N,)

    # pen2 = pen.unsqueeze(1)                      # (N,1)
    # # debugging assertion (remove or guard in production)
    # assert pen2.ndim == 2 and pen2.shape[1] == 1, f"Penalty term wrong shape: {pen2.shape}"
    # return pen2
    return pen


def r_yaw_stability(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward yaw stability (penalize yaw rate)."""
    ang_vel_b = env.scene[asset_cfg.name].data.root_ang_vel_b  # (N,3)
    yaw_rate = ang_vel_b[:, 2]  # z-component in body frame
    reward = -torch.square(yaw_rate)  # penalize large yaw rates
    return reward


def r_yaw_stability_if_horizontal(env, asset_cfg=SceneEntityCfg("robot"), horiz_thresh=0.1) -> torch.Tensor:
    """Reward yaw stability only if horizontal (small roll/pitch)."""
    quat = env.scene[asset_cfg.name].data.root_quat_w  # (N,4)
    # convert to roll, pitch, yaw
    #rpy = math_utils.euler_xyz_from_quat(quat)  # (N,3)
    (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(quat)  # tuple of (N,), (N,), (N,)

    ang_vel_b = env.scene[asset_cfg.name].data.root_ang_vel_b
    yaw_rate = ang_vel_b[:, 2]

    # mask: 1 if roll/pitch small, else 0
    horiz_mask = ((roll.abs() < horiz_thresh) & (pitch.abs() < horiz_thresh)).float()

    reward = -torch.square(yaw_rate) * horiz_mask
    return reward


def r_success_bonus(env, asset_cfg=SceneEntityCfg("robot"), reward_value=100.0) -> torch.Tensor:
    """Sparse success reward: big bonus when landed low (<0.1) and episode ends due to t_stuck_altitude."""
    asset = env.scene[asset_cfg.name]
    pos_z = asset.data.root_pos_w[:, 2]  # altitude (N,)

    # termination mask for this episode step
    stuck_mask = env.termination_manager.get_term("t_stuck_altitude").terminated  # (N,) bool

    # success = low altitude + stuck condition
    success = (pos_z < 0.1) & stuck_mask

    # only give reward at those final steps
    reward = torch.where(success, torch.full_like(pos_z, reward_value), torch.zeros_like(pos_z))
    return reward.unsqueeze(1)
