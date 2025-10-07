from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class OriginGoalCommandCfg(CommandTermCfg):
    """Config for a constant goal at origin (0,0,0)."""
    dim: int = 3              # number of command dimensions
    asset_name: str = "robot"

class OriginGoalCommand(CommandTerm):
    """Concrete command term: constant goal (origin)."""
    cfg: OriginGoalCommandCfg

    def __init__(self, cfg: OriginGoalCommandCfg, env):
        super().__init__(cfg, env)
        self._device = env.device
        # (num_envs, dim)
        self._goal = torch.zeros((env.num_envs, cfg.dim), device=self._device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Required abstract method implementations
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor):
        # Nothing to randomize; keep origin. Ensure buffer shape remains valid.
        if env_ids.numel() > 0:
            self._goal[env_ids] = 0.0

    def _update_command(self):
        # Command does not change each step (static origin)
        pass

    def _update_metrics(self):
        # No metrics to accumulate for a static command
        pass

    # ------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        # Manager expects (num_envs, dim)
        return self._goal

    @property
    def num_commands(self) -> int:
        return self.cfg.dim

    # Optional: allow direct call
    def __call__(self, env, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self._goal
        return self._goal[env_ids]