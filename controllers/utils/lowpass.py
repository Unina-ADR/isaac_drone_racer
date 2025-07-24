from dataclasses import dataclass

import torch


@dataclass
class LowPassFilterParams:
    device: str = "cuda"
    dim: torch.Size = torch.Size([64,1])
    dt: float = 0.001
    cutoff_freq: float = 100.0
    initial_value: float = 0.0


class LowPassFilter:
    def __init__(self, params: LowPassFilterParams):
        self._params = params
        self._all_env_id = torch.arange(params.dim[0], device=params.device)

        self._alpha = 1 / (1 + 1/ (2 * torch.pi * params.cutoff_freq * params.dt))
        self._output = params.initial_value * torch.ones(self._params.dim, device=params.device)

    def reset(self, env_ids: torch.Tensor = None, initial_value: float = None):
        if env_ids is None:
            env_ids = self._all_env_id
        
        if initial_value is None:
            initial_value = self._params.initial_value

        self._output[env_ids, ...] = initial_value

    def get_output(self) -> torch.Tensor:
        return self._output
    
    def update(self, data: torch.Tensor):
        self._output += self._alpha * (data - self._output)