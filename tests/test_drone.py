# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script spawns a quadcopter and idles the simulation (no controls).

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script spawns a quadcopter and idles the simulation (no controls).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from assets.a2r_drone import A2R_DRONE  # isort:skip


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a minimal quadcopter scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = A2R_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.03, 0.0, 0.0654), rot=(-0.3535533905932737, 0.3535533905932737, -0.616123724356957945, 0.6123724356957945), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=640,
        height=480,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs a passive simulation loop (no control, no periodic reset)."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # One-time placement of robot root at environment origins.
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    scene.reset()
    print("[INFO]: Drone spawned. Running passive simulation...")

    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete. Spawning drone only...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()