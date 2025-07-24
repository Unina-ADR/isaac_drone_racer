import gymnasium as gym

from . import agents

gym.register(
    id="My-Drone-Racer-v0",
    entry_point="isaaclab.envs:DirectRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.DroneRacerDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)