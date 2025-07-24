from __future__ import annotations



from assets.a2r_drone import A2R_DRONE

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from drone_racer.track_generator import generate_track


@configclass
class RacerDirectSceneCfg(InteractiveSceneCfg):
    """
    Configuration for the Racer Direct Scene.
    """
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

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

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )