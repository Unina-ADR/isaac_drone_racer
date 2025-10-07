# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, GaussianNoiseCfg, UniformNoiseCfg
import isaaclab.utils.math as math_utils
from . import mdp
from .track_generator import generate_track

from assets.a2r_drone import A2R_DRONE  # isort:skip

def deg2rad(deg):
    return deg * torch.pi / 180.0


@configclass
class DroneRacerSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # track
    track: RigidObjectCollectionCfg = generate_track(
        track_config = {
            "1":  {"pos": (12.5, 2.0, 1.45), "yaw":         deg2rad(180.0)},
            "2":  {"pos": (6.5,  6.0, 1.45), "yaw":  deg2rad(180.0 - 45.0)},
            "3":  {"pos": (5.5, 14.0, 1.45), "yaw":         deg2rad(150.0)},
            "4":  {"pos": (2.5, 24.0, 1.45), "yaw":          deg2rad(90.0)},
            "5":  {"pos": (7.5, 30.0, 1.45), "yaw":         deg2rad(-10.0)},
            "6":  {"pos": (12.2,22.0, 1.45), "yaw":                    0.0},
            "7":  {"pos": (17.5, 30, 4.15),  "yaw":          deg2rad(80.0)},
            "8":  {"pos": (17.5, 30, 1.45),  "yaw":         deg2rad(260.0)}, #7, 8 splits
            "9":  {"pos": (18.5, 22.0, 1.45),"yaw":         deg2rad(-80.0)},
            "10": {"pos": (20.5, 14.0, 1.45),"yaw":        deg2rad(-100.0)},
            "11": {"pos": (18.5, 6.0, 4.15), "yaw":    deg2rad(180 + 45.0)},
            "12": {"pos": (18.5, 6.0, 1.45), "yaw":    deg2rad(180 + 45.0)}, #11, 12 corkscrew
        }
    )

    # robot
    robot: ArticulationCfg = A2R_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    collision_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=True)
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/body", debug_vis=False)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.03, 0.0, 0.0654), rot=(0.0, 0.0, -0.86603, 0.5)), #isaac quat = (w x y z)
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=640,
        height=480,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False, debug_vis = True)



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        position = ObsTerm(func=mdp.root_pos_w, params={"pos_max": 30.0}, 
                           noise=GaussianNoiseCfg(std=0.05, mean = 0.0), scale = 1/30.0
        )
        attitude = ObsTerm(func=mdp.root_quat_w,
                           noise=GaussianNoiseCfg(std=0.01, mean = 0.0)
        )
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b, noise=GaussianNoiseCfg(std=0.01, mean = 0.0), scale = 1/5.0
        )
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b, params={"ang_vel_max": 11.69}, noise=GaussianNoiseCfg(std=0.01, mean = 0.0), scale = 1/11.69
        )
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target", "pos_max": 30.0})
        actions = ObsTerm(func=mdp.action_obs)
        waypoint = ObsTerm(func=mdp.waypoint_obs, params={"command_name": "target"})#, noise=GaussianNoiseCfg(std=0.05, mean = 0.0)
        #)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        image = ObsTerm(func=mdp.image)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise = GaussianNoiseCfg(std=0.01, mean = 0.0))
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc, noise = GaussianNoiseCfg(std=0.01, mean = 0.0))
        imu_att = ObsTerm(func=mdp.imu_orientation, noise = GaussianNoiseCfg(std=0.01, mean = 0.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # TODO: Resetting base happens in the command reset also for the moment
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.5, -1.5),
                "y": (-0.5, 0.5),
                "z": (1.5, 0.5),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0), 
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "mass_distribution_params": (0.95, 1.05),  # uniform distribution,
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # intervals
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.2),
    #     params={
    #         "force_range": (-0.1, 0.1),
    #         "torque_range": (-0.05, 0.05),
    #     },
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target = mdp.GateTargetingCommandCfg(
        asset_name="robot",
        track_name="track",
        randomise_start=None,
        record_fpv=False,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    #NB: Ho scalato 20x tutti i reward per rispettare l'implementazione originale di isaac_drone_racer
    
    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    
    progress = RewTerm(func=mdp.progress, weight=30.0, params={"command_name": "target"})
    
    gate_passed = RewTerm(func=mdp.gate_passed, weight=400.0, params={"command_name": "target"})
    
    lookat_next = RewTerm(func=mdp.lookat_next_gate, weight=0.20, params={"command_name": "target", "std": -10.0})
    
    action_reward = RewTerm(func=mdp.action_reward, weight=1, params={"weight_omega": -0.0002, "weight_rate": -0.0001})
    
    linear_vel_forward = RewTerm(func=mdp.linear_vel_forward, weight=-0.02, params={})
    
    linear_vel_side = RewTerm(func=mdp.linear_vel_side, weight=-0.01, params={})

    # lin_vel_to_next_gate= RewTerm(
    #     func=mdp.lin_vel_to_next_gate, weight=-0.01, params={"command_name": "target"}
    # )
    # time_reward = RewTerm(
    #     func=mdp.time_reward, weight=-0.01, params={}
    # )
    guidance_reward = RewTerm(
        func=mdp.guidance_reward, weight=-1.0, params={"command_name": "target"}
    )

    #TO DO: Add a reward to encourage using yaw to turn towards next gate
    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 20.0})
    collision = DoneTerm(
        func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.0001}
    )
    dynamic_limits_exceeded = DoneTerm(func=mdp.dynamic_limits_exceeded, params={"linvel_max": 5.0, "angvel_max": 11.69})
    walls = DoneTerm(func=mdp.walls, params={"min_x": 0.0, "min_y": 0.0, "max_x": 25.0, "max_y": 35.0})

    # out_of_bounds = DoneTerm(
    #     func=mdp.out_of_bounds,
    #     params={"bounds": (6.0, 8.0), "asset_cfg": SceneEntityCfg("robot")} #y_max, z_max 
    # )


@configclass
class DroneRacerEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable IMU and Tiled Camera
        self.scene.imu = None
        self.scene.tiled_camera = None

        # MDP settings
        self.observations.critic = None
        self.events.reset_base = None
        self.commands.target.randomise_start = True

        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable IMU and Tiled Camera
        self.scene.imu = None
        self.scene.tiled_camera = None

        # MDP settings
        self.observations.critic = None
        #self.events.reset_base = None
        #self.commands.target.randomise_start = None

        

        # Disable push robot events
        self.events.push_robot = None

        # Enable recording fpv footage
        #self.commands.target.record_fpv = True

        yaw =torch.tensor(deg2rad(180.0))
        roll= torch.tensor(deg2rad(0.0))
        pitch = torch.tensor(deg2rad(0.0))

        self.scene.robot.init_state.pos = (18.5, 2.0, 0.2)   # 5m high
        self.scene.robot.init_state.rot = math_utils.quat_from_euler_xyz(roll,pitch,yaw)
        self.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)

        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
