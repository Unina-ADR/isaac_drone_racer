from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def t_stuck_altitude(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window: int = 50,
    threshold: float = 0.02,
):
    """
    Terminate if altitude variation is too small over the last N steps.
    
    Args:
        window: number of steps to look back
        threshold: max altitude std dev (m) to be considered "stuck"
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]  # (num_envs,)

    # keep buffer in env (per-env altitude history)
    if not hasattr(env, "_alt_history"):
        env._alt_history = torch.zeros((env.num_envs, window), device=z.device)

    # roll & update buffer
    env._alt_history = torch.roll(env._alt_history, shifts=-1, dims=1)
    env._alt_history[:, -1] = z

    # compute std dev per env
    std = env._alt_history.std(dim=1)
    stuck = (std < threshold)
    return stuck
