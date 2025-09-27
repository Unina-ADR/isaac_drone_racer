"""
    Ported from ErcBunny/IsaacGymEnvs SimpleBetaflight implementation    
    https://github.com/ErcBunny/IsaacGymEnvs/tree/main

    
"""



import torch
from typing import List, Tuple


from dataclasses import dataclass, field
#from dynamics import Allocation
from .utils.lowpass import LowPassFilter, LowPassFilterParams

@dataclass
class BetaflightControllerParams:
    device: str = "cuda"
    num_envs: int = 64
    dt: float = 1 / 500
    thrust_coeff: float = 1.52117969e-5
    drag_coeff: float = 7.6909678e-9
    center_sensitivity: List[float] = field(
        default_factory=lambda: [70.0, 70.0, 70.0]
    )
    
    max_rate: List[float] = field(default_factory=lambda: [670.0, 670.0, 670.0]) #deg/s
    rate_expo: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # PID (rad) for RPY
    kp: List[float] = field(default_factory=lambda: [45.0, 47.0, 45.0])
    ki: List[float] = field(default_factory=lambda: [80.0, 84.0, 80.0])
    kd: List[float] = field(default_factory=lambda: [30.0, 34.0, 0.0])
    kff: List[float] = field(default_factory=lambda: [120.0, 125.0, 120.0])
    iterm_lim: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    pid_sum_lim: List[float] = field(default_factory=lambda: [1000.0, 1000.0, 1000.0])

    # d-term low pass filter cutoff frequency in Hz
    dterm_lpf_cutoff: float = 100

    # rotor positions in body FRD frame
    # all rotors are assumed to only produce thrust along the body-z axis
    # so z component does not matter anyway
    # rotor indexing: https://betaflight.com/docs/wiki/configurator/motors-tab
    

    #Rotors are indexed following betaflight convention (frame is FRD): 
    # 1: back right, 2: front right, 3:back left, 4: front left
    # rotor positions in body frame (x, y, z) in meters
    
    #ERCBUNNY POSITIONS
    # rotors_x: List[float] = field(
    #     default_factory=lambda: [-0.078665, 0.078665, -0.078665, 0.078665]
    # )
    # rotors_y: List[float] = field(
    #     default_factory=lambda: [0.097143, 0.097143, -0.097143, -0.097143]
    # )
    rotors_x: List[float] = field(
        default_factory=lambda: [-0.122, 0.14825, -0.122, 0.14825]
    )
    rotors_y: List[float] = field(
        default_factory=lambda: [-0.1343, -0.10474, 0.1343, 0.10474]
    )
    rotors_dir: List[int] = field(default_factory=lambda: [1, -1, -1, 1])
    pid_sum_mixer_scale: float = 1000.0

    # output idle
    output_idle: float = 0.05

    # throttle boost
    throttle_boost_gain: float = 10.0
    throttle_boost_freq: float = 50.0

    # thrust linearization
    thrust_linearization_gain: float = 0.4


class BetaflightController:
    def __init__(self, params: BetaflightControllerParams):
        self.params = params
        self.all_env_ids = torch.arange(
            params.num_envs, device=params.device, dtype=torch.int32
        )

        #inputs
        self.command= torch.zeros(params.num_envs, 4, device=params.device)

        # rate
        self.center_sensitivity = torch.tensor(
            params.center_sensitivity, device=params.device
        )
        self.max_rate = torch.tensor(params.max_rate, device=params.device)
        self.rate_expo = torch.tensor(params.rate_expo, device=params.device)

        # PID parameters
        self.kp = torch.tensor(params.kp, device=params.device)
        self.ki = torch.tensor(params.ki, device=params.device)
        self.kd = torch.tensor(params.kd, device=params.device)
        self.kff = torch.tensor(params.kff, device=params.device)
        self.iterm_lim = torch.tensor(params.iterm_lim, device=params.device)
        self.pid_sum_lim = torch.tensor(params.pid_sum_lim, device=params.device)
        self.int_err_ang_vel = torch.zeros(params.num_envs, 3, device=params.device)
        self.last_ang_vel = torch.zeros(params.num_envs, 3, device=params.device)
        dterm_lpf_params = LowPassFilterParams()
        dterm_lpf_params.device = params.device
        dterm_lpf_params.dim = self.last_ang_vel.size()
        dterm_lpf_params.dt = params.dt
        dterm_lpf_params.cutoff_freq = params.dterm_lpf_cutoff
        dterm_lpf_params.initial_value = 0.0
        self.dterm_lpf = LowPassFilter(dterm_lpf_params)

        # mixing table
        if not (
            len(params.rotors_x) == len(params.rotors_y)
            and len(params.rotors_y) == len(params.rotors_dir)
        ):
            raise ValueError("Rotors configuration error.")
        self.num_rotors = len(params.rotors_x)
        rotors_x_abs = [abs(item) for item in params.rotors_x]
        rotors_y_abs = [abs(item) for item in params.rotors_y]
        scale = max(max(rotors_x_abs), max(rotors_y_abs))
        mix_table_data = []
        for i in range(self.num_rotors):
            mix_table_data.append(
                [
                    1,  # throttle
                    -params.rotors_y[i] / scale,  # roll
                    params.rotors_x[i] / scale,  # pitch
                    -params.rotors_dir[i],  # yaw
                ]
            )
        self.mix_table = torch.tensor(mix_table_data, device=params.device)

        # throttle boost
        throttle_boost_lpf_params = LowPassFilterParams()
        throttle_boost_lpf_params.device = params.device
        throttle_boost_lpf_params.dim = torch.Size([params.num_envs])
        throttle_boost_lpf_params.dt = params.dt
        throttle_boost_lpf_params.cutoff_frequency = params.throttle_boost_freq
        throttle_boost_lpf_params.initial_value = 0.0
        self.throttle_boost_lpf = LowPassFilter(throttle_boost_lpf_params)

        # thrust linearization
        self.thrust_linearization_throttle_compensation = (
            params.thrust_linearization_gain - 0.5 * params.thrust_linearization_gain**2
        )

    def reset(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids=self.all_env_ids
            
        self.int_err_ang_vel[env_ids, ...] = 0.0
        self.last_ang_vel[env_ids, ...] = 0.0
        self.dterm_lpf.reset(env_ids)
        self.throttle_boost_lpf.reset(env_ids)



    def set_command(self, command: torch.Tensor):
        self.command[:] = command

    def compute(self, ang_vel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the motor commands based on the angular velocity and the set command.
        
        :param ang_vel: Tensor of shape (num_envs, 3) representing the angular velocity in rad/s.
        :return: Tuple of tensors (motor_commands, throttle_boost).
        """

        des_ang_vel = _compute_input_map_script(
            command=self.command,
            center_sensitivity=self.center_sensitivity,
            max_rate=self.max_rate,
            rate_expo=self.rate_expo,
        )

        err_ang_vel = des_ang_vel - ang_vel

        self.int_err_ang_vel += err_ang_vel
        self.int_err_ang_vel.clamp_(
            min=-self.iterm_lim, max=self.iterm_lim
        )
        # self.int_err_ang_vel = torch.clamp(
        #     self.int_err_ang_vel, -self.iterm_lim, self.iterm_lim
        # )

        d_ang_vel=self.dterm_lpf.get_output()

        pid_sum = _compute_pid_sum_script(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            kff=self.kff,
            pid_sum_lim=self.pid_sum_lim,
            pid_sum_mixer_scale=self.params.pid_sum_mixer_scale,
            err_ang_vel=err_ang_vel,
            int_err_ang_vel=self.int_err_ang_vel,
            d_ang_vel=d_ang_vel,
            des_ang_vel=des_ang_vel,
        )

        self.dterm_lpf.update((ang_vel - self.last_ang_vel) / self.params.dt)
        self.last_ang_vel[:] = ang_vel

        cmd_t = (self.command[:, 2] + 1.0) / 2.0
        throttle_low_freq_component = self.throttle_boost_lpf.get_output()

        u=_compute_mixing_script(
            mix_table=self.mix_table,
            throttle_boost_gain=self.params.throttle_boost_gain,
            thrust_linearization_throttle_compensation=self.thrust_linearization_throttle_compensation,
            thrust_linearization_gain=self.params.thrust_linearization_gain,
            output_idle=self.params.output_idle,
            pid_sum=pid_sum,
            cmd_t=cmd_t,
            throttle_low_freq_component=throttle_low_freq_component,
        )

        self.throttle_boost_lpf.update(cmd_t)

        return des_ang_vel, u
    
    def get_thrust_and_torque_command(self, omega_max, thrust_coeff, drag_coeff, u) -> torch.Tensor:
        """
        Computes the total thrust and body torques given the motor outputs.

        Parameters:
        - thrust_coeff (float): Coefficient for thrust calculation.
        - drag_coeff (float): Coefficient for drag calculation.
        - u (torch.Tensor): Motor outputs of shape (num_envs, 4).

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4) representing total thrust and body torques.
        """
        # Provisional mixing table: drag_coeff for thrust, roll, pitch; drag_coeff for yaw
        # Shape: [num_rotors, 4]
        provisional_mix_table = self.mix_table.clone()
        provisional_mix_table[:, 0:3] *= thrust_coeff  # thrust, roll, pitch columns
        provisional_mix_table[:, 3] *= drag_coeff    # yaw column

        # Compute rotor angular velocities from motor outputs
        omega_sq = (u * omega_max)**2
        # Compute thrusts from rotor angular velocities
        # Compute total thrust and body torques using the provisional mixing table
        thrust_torque = torch.matmul(provisional_mix_table, omega_sq.T).T
        return thrust_torque
    


@torch.jit.script
def _compute_input_map_script(
    command: torch.Tensor,
    center_sensitivity: torch.Tensor,
    max_rate: torch.Tensor,
    rate_expo: torch.Tensor,
) -> torch.Tensor:
    """
    Maps stick positions to desired body angular velocity:
    https://betaflight.com/docs/wiki/guides/current/Rate-Calculator.

    Assuming FRD body frame:
    channel A -> roll (body x),
    channel E -> pitch (body y),
    channel R -> yaw (body z).

    Let x[-1, 1] be the stick position, d the center sensitivity, f the max rate, g the expo,
    desired body rate = sgn(x) * ( d|x| + (f-d) * ( (1-g)x^2 + gx^6 ) )
    """
    # DEBUG PRINTS
    # print("[DEBUG] _compute_input_map_script: command shape:", command.shape)
    # print("[DEBUG] _compute_input_map_script: center_sensitivity:", center_sensitivity)
    # print("[DEBUG] _compute_input_map_script: max_rate:", max_rate)
    # print("[DEBUG] _compute_input_map_script: rate_expo:", rate_expo)
    cmd_aer = command[:, [0, 1, 3]]
    #print("[DEBUG] _compute_input_map_script: cmd_aer shape:", cmd_aer.shape)
    des_body_rates = torch.sgn(cmd_aer) * (
        center_sensitivity * torch.abs(cmd_aer)
        + (max_rate - center_sensitivity)
        * ((1 - rate_expo) * torch.pow(cmd_aer, 2) + rate_expo * torch.pow(cmd_aer, 6))
    )
    #print("[DEBUG] _compute_input_map_script: des_body_rates (deg/s):", des_body_rates)
    des_body_rates_rad = torch.deg2rad(des_body_rates)
    #print("[DEBUG] _compute_input_map_script: des_body_rates (rad/s):", des_body_rates_rad)
    return des_body_rates_rad
    

@torch.jit.script
def _compute_pid_sum_script(
    kp: torch.Tensor,
    ki: torch.Tensor,
    kd: torch.Tensor,
    kff: torch.Tensor,
    pid_sum_lim: torch.Tensor,
    pid_sum_mixer_scale: float,
    err_ang_vel: torch.Tensor,
    int_err_ang_vel: torch.Tensor,
    d_ang_vel: torch.Tensor,
    des_ang_vel: torch.Tensor,
) -> torch.Tensor:
    # DEBUG PRINTS
    # print("[DEBUG] _compute_pid_sum_script: kp:", kp)
    # print("[DEBUG] _compute_pid_sum_script: ki:", ki)
    # print("[DEBUG] _compute_pid_sum_script: kd:", kd)
    # print("[DEBUG] _compute_pid_sum_script: kff:", kff)
    # print("[DEBUG] _compute_pid_sum_script: pid_sum_lim:", pid_sum_lim)
    # print("[DEBUG] _compute_pid_sum_script: pid_sum_mixer_scale:", pid_sum_mixer_scale)
    # print("[DEBUG] _compute_pid_sum_script: err_ang_vel shape:", err_ang_vel.shape)
    # print("[DEBUG] _compute_pid_sum_script: int_err_ang_vel shape:", int_err_ang_vel.shape)
    # print("[DEBUG] _compute_pid_sum_script: d_ang_vel shape:", d_ang_vel.shape)
    # print("[DEBUG] _compute_pid_sum_script: des_ang_vel shape:", des_ang_vel.shape)
    pid_sum = (
        kp * err_ang_vel + ki * int_err_ang_vel - kd * d_ang_vel + kff * des_ang_vel
    )
    #print("[DEBUG] _compute_pid_sum_script: pid_sum (before clamp):", pid_sum)
    pid_sum.clamp_(min=-pid_sum_lim, max=pid_sum_lim)
    #print("[DEBUG] _compute_pid_sum_script: pid_sum (after clamp):", pid_sum)
    # scale the PID sum before mixing
    pid_sum /= pid_sum_mixer_scale
    #print("[DEBUG] _compute_pid_sum_script: pid_sum (after scale):", pid_sum)
    return pid_sum

@torch.jit.script
def _compute_mixing_script(
    mix_table: torch.Tensor,
    throttle_boost_gain: float,
    thrust_linearization_throttle_compensation: float,
    thrust_linearization_gain: float,
    output_idle: float,
    pid_sum: torch.Tensor,
    cmd_t: torch.Tensor,
    throttle_low_freq_component: torch.Tensor,
):
    # DEBUG PRINTS
    #print("[DEBUG] _compute_mixing_script: mix_table shape:", mix_table.shape)
    #print("[DEBUG] _compute_mixing_script: pid_sum shape:", pid_sum.shape)
    #print("[DEBUG] _compute_mixing_script: cmd_t shape:", cmd_t.shape)
    #print("[DEBUG] _compute_mixing_script: throttle_low_freq_component shape:", throttle_low_freq_component.shape)
    rpy_u = torch.matmul(mix_table[:, 1:], pid_sum.T).T
    #print("[DEBUG] _compute_mixing_script: rpy_u shape:", rpy_u.shape)
    #print("[DEBUG] _compute_mixing_script: rpy_u:", rpy_u)
    rpy_u_max = torch.max(rpy_u, 1).values
    rpy_u_min = torch.min(rpy_u, 1).values
    rpy_u_range = rpy_u_max - rpy_u_min
    #print("[DEBUG] _compute_mixing_script: rpy_u_max:", rpy_u_max)
    #print("[DEBUG] _compute_mixing_script: rpy_u_min:", rpy_u_min)
    #print("[DEBUG] _compute_mixing_script: rpy_u_range:", rpy_u_range)
    norm_factor = 1/ rpy_u_range
    norm_factor.clamp_(max=1.0)
    #print("[DEBUG] _compute_mixing_script: norm_factor:", norm_factor)
    rpy_u_normalized = norm_factor.view(-1, 1) * rpy_u
    rpy_u_normalized_max = norm_factor * rpy_u_max
    rpy_u_normalized_min = norm_factor * rpy_u_min
    #print("[DEBUG] _compute_mixing_script: rpy_u_normalized shape:", rpy_u_normalized.shape)
    throttle_high_freq_component = cmd_t - throttle_low_freq_component
    #print("[DEBUG] _compute_mixing_script: throttle_high_freq_component:", throttle_high_freq_component)
    throttle = cmd_t + throttle_boost_gain * throttle_high_freq_component
    #print("[DEBUG] _compute_mixing_script: throttle (before clamp):", throttle)
    throttle.clamp_(min=0.0, max=1.0)
    #print("[DEBUG] _compute_mixing_script: throttle (after clamp):", throttle)
    throttle /= 1 + thrust_linearization_throttle_compensation * torch.pow(
        1 - throttle, 2
    )
    #print("[DEBUG] _compute_mixing_script: throttle (after linearization step 1):", throttle)
    throttle.clamp_(min=-rpy_u_normalized_min, max=(1 - rpy_u_normalized_max))
    #print("[DEBUG] _compute_mixing_script: throttle (after constrain):", throttle)
    u_rpy_t = rpy_u_normalized + throttle.view(-1, 1)
    #print("[DEBUG] _compute_mixing_script: u_rpy_t shape:", u_rpy_t.shape)
    u_rpy_t *= 1 + thrust_linearization_gain * torch.pow(1 - u_rpy_t, 2)
    #print("[DEBUG] _compute_mixing_script: u_rpy_t (after linearization step 2):", u_rpy_t)
    u = output_idle + (1 - output_idle) * u_rpy_t
    #print("[DEBUG] _compute_mixing_script: u (final output):", u)
    return u