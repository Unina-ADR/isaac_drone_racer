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


# Test spawn cube
import isaacsim.core.utils.prims as prim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

from assets.a2r_drone import A2R_DRONE  # isort:skip

def deg2rad(deg):
    return deg * torch.pi / 180.0


@configclass
class LanderSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the Lander task."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn = sim_utils.GroundPlaneCfg(),
    )

     # robot
    robot: ArticulationCfg = A2R_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    collision_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=True)
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/body", debug_vis=False)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.03, 0.0, 0.0654), rot=(0.0, 0.0, -0.86603, 0.5)),
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
    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model = True)

@configclass
class ObservationsCfg:

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
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b, noise=GaussianNoiseCfg(std=0.01, mean = 0.0), scale = 1/11.69
        )

        last_action = ObsTerm(func=mdp.action_obs)

        goal_pos_body = ObsTerm(func=mdp.goal_pos_in_body_frame, scale = 1/30.0)
        #dist_from_goal = ObsTerm(func=mdp.dist_from_goal_w)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:

    rew_distance = RewTerm(func = mdp.r_dist, weight = 15.0)#, params = {"target_pos": (0.0, 0.0, 0.0), "dist_max": 30.0}) 
    rew_up = RewTerm(func = mdp.r_upright, weight = 5.0)
    rew_vsoft = RewTerm(func = mdp.r_vsoft, weight = 3.0)
    rew_hspeed = RewTerm(func = mdp.r_hspeed, weight = -3.0)
    action_reward = RewTerm(func=mdp.action_reward, weight=1, params={"weight_omega": -0.0002, "weight_rate": -0.0001})
    rew_penalize_upward = RewTerm(func=mdp.r_penalize_upward, weight=-3.0, params={"use_squared": True})
    #r_yaw_stability = RewTerm(func=mdp.r_yaw_stability, weight=-0.5)
    r_yaw_stability = RewTerm(func=mdp.r_yaw_stability_if_horizontal, weight=-1.0)
    #r_success_landing = RewTerm(func=mdp.r_success_bonus, weight=50.0)
    

@configclass
class EventsCfg:

    reset_base = EventTerm(func = mdp.reset_root_state_uniform,
                            mode = "reset",
                            params = {
                                "pose_range":{
                                    "x": (-2.0, 2.0),
                                    "y": (-2.0, 2.0),
                                    "z": (1.0, 3.0),
                                    "roll": (deg2rad(-30.0), deg2rad(30.0)),
                                    "pitch": (deg2rad(-30.0), deg2rad(30.0)),
                                    "yaw": (-torch.pi, torch.pi),
                                },
                                "velocity_range":{
                                    "x": (-5.0,5.0),
                                    "y": (-5.0, 5.0),
                                    "z": (-1.0, 1.0),
                                    "roll": (-8.0, 8.0),
                                    "pitch": (-7.0, 7.0),
                                    "yaw": (-7.0, 7.0),
                                },

                            }
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "mass_distribution_params": (1.15, 1.35),  # uniform distribution,
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

@configclass
class CommandsCfg:
    goal: mdp.OriginGoalCommandCfg = mdp.OriginGoalCommandCfg(
        class_type = mdp.OriginGoalCommand,
        resampling_time_range=(1e9,1e9),
        debug_vis = False
    )



@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.0001})
    stabilized = DoneTerm(func=mdp.t_stuck_altitude, params={"asset_cfg": SceneEntityCfg("robot"), "window": 800, "threshold": 0.01})




@configclass
class LanderEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for the Lander task."""

    # scene
    scene: LanderSceneCfg = LanderSceneCfg(num_envs = 4096, env_spacing = 0.0)

    # managers
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable IMU and Tiled Camera
        self.scene.imu = None
        self.scene.tiled_camera = None

        # MDP settings
        #self.events.reset_base = None

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
class LanderEnvCfgPLAY(LanderEnvCfg):
    """
    Config for inference/play runs:
    - deterministic spawn (fixed initial state)
    - custom timeout conditions
    """
    # override init state
    def __post_init__(self):
        super().__post_init__()
        yaw =torch.tensor(deg2rad(0.0))
        roll= torch.tensor(deg2rad(-15.0))
        pitch = torch.tensor(deg2rad(25.0))
        # Example: spawn at fixed pos/rot instead of random
        self.scene.robot.init_state.pos = (2.0, 4.0, 1.0)   # 5m high
        self.scene.robot.init_state.rot = math_utils.quat_from_euler_xyz(roll,pitch,yaw)# (0.0, 0.0, 0.0, 1.0)  # no rotation
        self.scene.robot.init_state.lin_vel = (3.0, -2.0, 0.2)
        self.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
        
        # disable randomization in play mode
        self.observations.policy.randomization = None
