from __future__ import annotations

import math
import torch

from assets.a2r_drone import A2R_DRONE
from dynamics import Allocation, Motor
from controllers import BetaflightController, BetaflightControllerParams


from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg

 # Setup drone articulation
from assets.a2r_drone import A2R_DRONE
from drone_racer.track_generator import generate_track
from waypoint_manager.waypoint_tracker import WaypointTracker



@configclass
class DroneRacerDirectEnvCfg(DirectRLEnvCfg):
    # Environment parameters
    decimation = 4
    episode_length_s = 15.0
    action_scale = 1.0
    action_space = 4
    observation_space = 51

    # Reward scales (placeholders)
    rew_scale_progress = 1.0
    rew_scale_action_ang = -0.0002
    rew_scale_action_rate = -0.0001
    rew_scale_collision = -5.0
    rew_scale_lookat = 0.02
    rew_scale_exp = -10
    rew_scale_vel_horizontal = -0.01
    rew_scale_vel_forward = -0.02
    guidance_x_thresh = 3.0
    guidance_tol = 0.2
    k_rejection = 2.0

    #waypoint parameters
    wp_height = 1.5  # Height of the waypoints
    wp_width = 1.5  # Width of the waypoints

    # Drone parameters (placeholders)
    v_lin_max = 10.0
    v_ang_max = 10.0
    max_corner_dist = 20.0
    max_wp_dist = 15.0

    #reset parameters
    #reset_pos_range = [-1.0, 31.0]
    #reset_vel_range = [0.0, 1.0]
    #reset_ang_vel_range = [0.0, 0.0]
    reset_yaw = 0.0  # Fixed yaw for the drone
    reset_pos = (0.0, 0.0, 1.0)  # Initial position of the drone
    reset_vel = (0.0, 0.0, 0.0)  # Initial velocity of the drone



    # Scene and simulation configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(dt=0.004, render_interval=decimation)

    track_config={
        "1": {"pos": (1.0, 0.0, 1.5), "yaw": 0.0},
        "2": {"pos": (6.0, 2.5, 1.5), "yaw": 0.0},
        "3": {"pos": (11.0, -2.5, 1.5), "yaw": 0.0},
        "4": {"pos": (16.0, 2.5, 1.5), "yaw": 0.0},
        "6": {"pos": (26.0, 2.5, 1.5), "yaw": 0.0},
        "7": {"pos": (31.0, -2.5, 1.5), "yaw": 0.0},
    }
    

    # Robot and motor configuration (placeholders)
    robot_cfg = ArticulationCfg = A2R_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")
    m1_dof_name = "m1_joint"
    m2_dof_name = "m2_joint"
    m3_dof_name = "m3_joint"
    m4_dof_name = "m4_joint"
    thrust_coeff = 5.3453e-7
    drag_coeff = 1.5e-9

    # Controller configuration
    # controller_params: BetaflightControllerParams = BetaflightControllerParams(
    #     num_envs=4096, 
    # )
    

class DroneRacerDirectEnv(DirectRLEnv):
    cfg: DroneRacerDirectEnvCfg

    def __init__(self, cfg: DroneRacerDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # --- Variables ---
        self._drone_motor_idx = None
        self.action_scale = cfg.action_scale
        self.last_actions = None
        self.joint_pos = None
        self.joint_vel = None
        self.omega_max = 5541.0  # Maximum angular velocity of the drone motors in rad/s
        self._elapsed_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.controller_params = BetaflightControllerParams(num_envs=self.num_envs)
        self.controller = BetaflightController(params=self.controller_params)
        self._allocation = Allocation(
            num_envs=self.num_envs,
            thrust_coeff=cfg.thrust_coeff,
            drag_coeff=cfg.drag_coeff,
            device=self.device,
            dtype=torch.float32,  # Assuming float32 for actions
        )
        self.last_actions = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.last_positions = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        cam_roll = torch.tensor([0.0], device=self.device).expand(self.num_envs, 1)
        cam_pitch = torch.tensor([-0.87266], device=self.device).expand(self.num_envs, 1)
        cam_yaw = torch.tensor([0.0], device=self.device).expand(self.num_envs, 1)
        self.cam_quat=math_utils.quat_from_euler_xyz(cam_roll,cam_pitch,cam_yaw)

        # self.wp_height = self.cfg.wp_height  # Height of the waypoints
        # self.wp_width = self.cfg.wp_width  # Width of the waypoints

        self.track=generate_track(cfg.track_config)

        self.waypoint_tracker = WaypointTracker(num_envs=self.num_envs, track=self.track)





    def _setup_scene(self):
       
        self.drone = ArticulationCfg(self.cfg.robot_cfg)
        self.scene.articulations["Robot"] = self.drone

        
        self.scene.rigid_object_collections["Track"] = self.track
        # Initialize the waypoint tracker  
        self.waypoint_tracker.set_waypoint_data()
        # Add ground plane
        spawn_ground_plane(prim_path="World/Ground", cfg=GroundPlaneCfg())

        # Add sensors to the drone
        self.drone.add_sensor(ImuCfg(prim_path=self.drone.prim_path + "/Imu", name="Imu"))
        self.drone.add_sensor(ContactSensorCfg(prim_path=self.drone.prim_path + "/ContactSensor", name="ContactSensor"))
        self.drone.add_sensor(TiledCameraCfg(prim_path=self.drone.prim_path + "/Camera", name="Camera", width=256, height=256, fov=math.pi / 2))

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
        light_cfg.func("World/light", light_cfg)

        # Clone environments and filter collisions if on CPU
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        #save last position and actions for reward computation
        self.last_positions = self.drone.data.root_pos_w.clone()
        self.last_actions = self.actions.clone()
        #action clamping (for now actions are directly mapped to motor thrusts)
        raw_actions = actions * self.action_scale
        clamped = raw_actions.clamp_(0.0, 1.0)
        self.actions = clamped
        


    def _apply_actions(self):
        # Apply actions to the drone motors
        thrust_torque = self._allocation.compute(self.actions)
        thrust = thrust_torque[:, 0]  # Extract thrust component
        moment = thrust_torque[:, 1:]
        self.drone.set_external_force_and_torque(thrust, moment, body_ids=self._body_id)
        self._elapsed_time += self.sim.dt

    def _get_observations(self):
        obs= _compute_observations_script(
            self.cfg.max_corner_dist,
            self.cfg.v_lin_max,
            self.cfg.v_ang_max,
            self.drone.data,
            self.last_actions,
            self.wp_height,
            self.wp_width,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self):
        total_reward = _compute_reward_script(
            self.cfg.rew_scale_progress,
            self.cfg.rew_scale_action_ang,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_collision,
            self.cfg.rew_scale_lookat,
            self.cfg.rew_scale_exp,
            self.cfg.rew_scale_vel_horizontal,
            self.cfg.rew_scale_vel_forward,
            self.drone.data,
            self.actions,
            self.last_actions,
            self.last_positions,
            self.target_wp,
            self.cam_quat,
            self.cfg.guidance_x_thresh,
            self.cfg.guidance_tol,
            self.cfg.k_rejection,
            self.cfg.wp_height,
            self.cfg.wp_width,
            self._elapsed_time,
        )
        return total_reward
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >=self.max_episode_length -1
        out_of_bounds = torch.logical_or(self.drone.data.root_pos_w[:,2]<0.1, self.drone.data.root_pos_w[:,2]>10.0) 
        # Reset environment state
        return time_out, out_of_bounds

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset drone position and velocity
        drone_pos = torch.tensor(self.cfg.reset_pos, device=self.device, dtype=torch.float32).expand(len(env_ids), 3)
        drone_vel = torch.tensor(self.cfg.reset_vel, device=self.device, dtype=torch.float32).expand(len(env_ids), 3)
        drone_ang_vel = torch.zeros_like(drone_vel)

        self.drone.write_root_pose_to_sim(drone_pos, env_ids)
        self.drone.write_root_vel_to_sim(drone_vel, env_ids)
        self.drone.write_root_ang_vel_to_sim(drone_ang_vel, env_ids)
        
        # Reset drone joint positions and velocities
        self.joint_pos = torch.zeros((len(env_ids), 4), device=self.device, dtype=torch.float32)
        self.joint_vel = torch.zeros((len(env_ids), 4), device=self.device, dtype=torch.float32)
        self.drone.write_joint_pos_to_sim(self.joint_pos, env_ids)
        self.drone.write_joint_vel_to_sim(self.joint_vel, env_ids)






@torch.jit.script
def _compute_reward_script(
    rew_scale_progress: float,
    rew_scale_action_ang: float,
    rew_scale_action_rate: float,
    rew_scale_collision: float,
    rew_scale_lookat: float,
    rew_scale_exp: float,
    rew_scale_vel_horizontal: float,
    rew_scale_vel_forward: float,
    drone_data: torch.Tensor,  # Placeholder for drone data
    actions: torch.Tensor,  # Placeholder for actions
    last_actions: torch.Tensor,  # Placeholder for last actions
    last_positions: torch.Tensor,  # Placeholder for last positions
    target_wp: torch.Tensor,  # Placeholder for target waypoints
    cam_quat: torch.Tensor,  # Placeholder for camera quaternion
    guidance_x_thresh: float,
    guidance_tol: float,
    k_rejection: float,
    wp_height: float,
    wp_width: float,
    elapsed_time: torch.Tensor,  # Placeholder for elapsed time
) -> torch.Tensor:
    """    Compute the reward based on the provided parameters and drone data.
        Progress reward
        action reward
        collision penalty
        lookat next gate reward
        velocity rewards
        guidance reward
        time reward
    """
    drone_pos_w = drone_data.root_pos_w  # Assuming root_pos_w is a tensor of shape (num_envs, 3)
    drone_quat_w = drone_data.root_quat_w
    drone_linvel_w = drone_data.root_vel_w

    #Progress reward
    old_distance = torch.norm(last_positions - target_wp, dim=-1)
    new_distance = torch.norm(drone_pos_w - target_wp, dim=-1)
    r_prog = rew_scale_progress * (old_distance - new_distance)

    #Action reward
    action_omega = actions[:, 1:4]
    action_rate = torch.square(torch.norm(actions - last_actions, dim=1))

    r_action_ang = rew_scale_action_ang * torch.norm(action_omega, dim=1)
    r_action_rate = rew_scale_action_rate * action_rate
    r_action = r_action_ang + r_action_rate

    # Look at next gate reward
    vec_to_gate = target_wp - drone_pos_w
    vec_to_gate = math_utils.normalize(vec_to_gate)
    
    
    drone_x_axis = math_utils.quat_apply(drone_quat_w, torch.tensor([1.0, 0.0, 0.0], device=drone_data.root_pos_w.device).expand(drone_quat_w.shape[0], 3))
    drone_x_axis = math_utils.normalize(drone_x_axis)
    cam_x_axis = math_utils.quat_apply(cam_quat, drone_x_axis)
    cam_x_axis = math_utils.normalize(cam_x_axis)
    
    dot = (cam_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    expangle = torch.pow(angle, 4)
    r_lookat = rew_scale_lookat * torch.exp(-rew_scale_exp*expangle)

    # Velocity rewards
    v_x = drone_linvel_w[:, 0]
    v_y = drone_linvel_w[:, 1]
    rew_velocity= rew_scale_vel_forward*torch.square(torch.min(v_x, torch.zeros_like(v_x))) + rew_scale_vel_horizontal*torch.square(v_y)

    #Collision penalty
    collision_penalty = torch.zeros_like(drone_data.root_pos_w[:, 0])  # Placeholder for collision penalty
    collision_penalty = (drone_data.sensor_data["ContactSensor"].read() | drone_pos_w[:, 2] < 0.0).float() * rew_scale_collision
    collision_penalty.unsqueeze(1)
    #guidance reward



    # time reward



    return torch.tensor(0.0, device=drone_data.root_pos_w.device, dtype=torch.float32)

@torch.jit.script
def _compute_observations_script(
            max_corner_dist: float,
            v_lin_max: float,
            v_ang_max: float,
            drone_data: torch.Tensor,  # Placeholder for drone data
            last_actions: torch.Tensor,  # Placeholder for last actions
            wp_height: float,
    ) -> torch.Tensor:
    """Compute the observations for the drone racer environment."""
    drone_pos_w = drone_data.root_pos_w  # Assuming root_pos_w is a tensor of shape (num_envs, 3)
    drone_quat_w = drone_data.root_quat_w
    drone_linvel_w = drone_data.root_vel_w
    drone_ang_vel_b = drone_data.root_ang_vel_b


    num_envs = drone_pos_w.shape[0]
    # [4, 3] corners in gate frame
    relative_gate_corners = torch.tensor(
        [
            [0.0, -wp_height/2, -wp_height/2],
            [0.0, -wp_height/2,  wp_height/2],
            [0.0, wp_height/2,   wp_height/2],
            [0.0, wp_height/2,  -wp_height/2],
        ],
        dtype=drone_pos_w.dtype,
        device=drone_pos_w.device,
    ).unsqueeze(0).expand(num_envs, -1, -1)  # [num_envs, 4, 3]

    # Flatten for batch processing
    relative_gate_corners_flat = relative_gate_corners.reshape(-1, 3)  # [num_envs*4, 3]
    next_gate_quat_expand = next_gate_quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)  # [num_envs*4, 4]
    gate_corners_w_flat = math_utils.quat_apply(next_gate_quat_expand, relative_gate_corners_flat)  # [num_envs*4, 3]
    gate_corners_w = gate_corners_w_flat.reshape(num_envs, 4, 3) + next_gate_pos.unsqueeze(1)  # [num_envs, 4, 3]

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
    s_c = (dot_product / (norm1 * norm2 + 1e-8)).unsqueeze(1)  # [num_envs, 1]

    # --- Compute corner vectors and norms ---
    corner_vectors = gate_corners_w - drone_pos.unsqueeze(1)  # [num_envs, 4, 3]
    vector_norms = torch.norm(corner_vectors, dim=-1)  # [num_envs, 4]
    corner_vectors_flat = corner_vectors.reshape(num_envs, -1)  # [num_envs, 12]
    vector_norms_scaled = torch.min(vector_norms/l_max, torch.ones_like(vector_norms))  # Scale norms to [0, 1]

    # --- Concatenate ---
    waypoint_obs = torch.cat([s_c, vector_norms_scaled, corner_vectors_flat], dim=1)  # [num_envs, 17]
    # Placeholder for observation computation logic
    # This should compute the actual observations based on the drone's state and actions
    observations = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device, dtype=torch.float32)