from __future__ import annotations

import math
import torch

from assets.a2r_drone import A2R_DRONE
from dynamics import Allocation, Motor
from controllers import BetaflightController, BetaflightControllerParams


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg

from .scene_cfg import RacerDirectSceneCfg
from drone_racer.track_generator import generate_track



@configclass
class RacerDirectEnvCfg(DirectRLEnvCfg):
    """
    Configuration for the Racer Direct Environment.
    """
    #env
    decimation = 4
    episode_length_s = 15.0
    action_scale = 1.0 # scale for the action space, 1.0 means no scaling
    action_space = 4
    observation_space = 51 # 3 pos, 4 quat, 3 vel, 3 ang vel, 4 action, 34 waypoint


    #reward scales
    rew_scale_progress = 1.0
    rew_scale_action_ang = -0.0002
    rew_scale_action_rate = -0.0001
    rew_scale_collision = -5.0
    rew_scale_lookat = 0.02
    rew_scale_exp = -10
    rew_scale_vel_horizontal = -0.01
    rew_scale_vel_forward = -0.02

    #drone parameters
    v_lin_max = 10.0
    v_ang_max = 6.0
    max_corner_dist = 10.0

    # scene configuration
    scene: InteractiveSceneCfg = RacerDirectSceneCfg(num_envs=4096, replicate_physics=True)

    # simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt = 0.004,
        render_interval = decimation
    )

    robot_cfg: ArticulationCfg = A2R_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")
    m1_dof_name = "m1_joint"
    m2_dof_name = "m2_joint"
    m3_dof_name = "m3_joint"
    m4_dof_name = "m4_joint"
    
    thrust_coeff = 5.3453e-7 # thrust coefficient for the drone motors
    drag_coeff = 1.5e-9 # drag coefficient for the drone motors
    #todo: add reset params
    # self.BetaflightControllerParams = BetaflightControllerParams(
        
    # )

class RacerDirectEnv(DirectRLEnv):
    cfg: RacerDirectEnvCfg

    def __init__(self, cfg: RacerDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        self._drone_motor_idx, _ = self.drone.find_joints(self.cfg.m1_dof_name, self.cfg.m2_dof_name, self.cfg.m3_dof_name, self.cfg.m4_dof_name)
        self._drone_motor_idx = torch.tensor(self._drone_motor_idx, dtype=torch.int32, device=self.device)
        self.action_scale = cfg.action_scale

        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)

        self.joint_pos = self.drone.data.joint_pos
        self.joint_vel = self.drone.data.joint_vel

        

    def _setup_scene(self):
        self.drone = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="World/Ground", cfg=GroundPlaneCfg())

        # track
        track: RigidObjectCollectionCfg = generate_track(
            # track_config={
            #     "1": {"pos": (0.0, 0.0, 1.5), "yaw": 0.0},
            #     "2": {"pos": (10.0, 5.0, 1.5), "yaw": 0.0},
            #     "3": {"pos": (10.0, -5.0, 1.5), "yaw": (5 / 4) * torch.pi},
            #     "4": {"pos": (-5.0, -5.0, 4.0), "yaw": torch.pi},
            #     "5": {"pos": (-5.0, -5.0, 1.5), "yaw": 0.0},
            #     "6": {"pos": (5.0, 0.0, 1.5), "yaw": (1 / 2) * torch.pi},
            #     "7": {"pos": (0.0, 5.0, 1.5), "yaw": torch.pi},
            # }
            track_config={
                "1": {"pos": (1.0, 0.0, 1.5), "yaw": 0.0},
                "2": {"pos": (6.0, 2.5, 1.5), "yaw": 0.0},
                "3": {"pos": (11.0, -2.5, 1.5), "yaw": 0.0},
                "4": {"pos": (16.0, 2.5, 1.5), "yaw": 0.0},
                "5": {"pos": (21.0, -2.5, 1.5), "yaw": 0.0},
                "6": {"pos": (26.0, 2.5, 1.5), "yaw": 0.0},
                "7": {"pos": (31.0, -2.5, 1.5), "yaw": 0.0},
            }
        )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["Robot"] = self.drone
        self.scene.rigid_object_collections["Track"] = track

        # sensors
        self.drone.add_sensor(ImuCfg(prim_path=self.drone.prim_path + "/Imu", name="Imu"))
        self.drone.add_sensor(ContactSensorCfg(prim_path=self.drone.prim_path + "/ContactSensor", name="ContactSensor"))
        self.drone.add_sensor(TiledCameraCfg(prim_path=self.drone.prim_path + "/Camera", name="Camera", width=256, height=256, fov=math.pi / 4))
    
        light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
        light_cfg.func("World/light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_actions = self.actions.clone()
        self.actions = self.action_scale * actions.clone()


    def _apply_actions(self):
        # apply actions to the drone motors
        


    
    def _get_observations(self):
        observations = compute_observations(self)

        return observations
    
    def _get_reward(self):

        return total_reward
    


@torch.jit.script
def compute_observations(env: RacerDirectEnv):
    """
    Computes the observations for the Racer Direct Environment.
    """
    drone = env.drone
    data = drone.data

    actions = env.actions

    # position, orientation, velocity, angular velocity
    pos = data.root_state_w[:, :3]
    quat = data.root_state_w[:, 3:7]
    vel = data.root_state_w[:, 7:10]
    ang_vel = data.root_state_w[:, 10:13]


