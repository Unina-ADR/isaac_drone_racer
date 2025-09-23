# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--hover_throttle", type=float, default=-1.0, help="Throttle stick value in [-1,1] for hover test (mapped to (x+1)/2).")
parser.add_argument("--omega_max", type=float, default=1855.0, help="Maximum rotor angular speed (rad/s) corresponding to motor output = 1.0.")
parser.add_argument("--log_interval", type=int, default=200, help="Steps between log prints.")
parser.add_argument("--thrust_coeff", type=float, default=1.52117969e-5, help="Rotor thrust coefficient.")
parser.add_argument("--drag_coeff", type=float, default=7.6909678e-9, help="Rotor drag (torque) coefficient.")
parser.add_argument("--log_thrust_torque", action="store_true", help="Enable logging of aggregated thrust/torque estimates.")
parser.add_argument("--force_scale", type=float, default=1.0, help="Scaling factor applied to computed thrust & torques before application.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

##
# Pre-defined configs
##
from assets.a2r_drone import A2R_DRONE  # isort:skip
from dynamics import Allocation
from controllers.betaflight import BetaflightController, BetaflightControllerParams  # new import


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = A2R_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/body/camera",
    #     #offset=TiledCameraCfg.OffsetCfg(pos=(0.14, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(-0.03, 0.0, 0.0654), rot=( 0.64086, -0.29884, 0.29884, -0.64086), convention="ros"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     width=640,
    #     height=480,
    # )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller: BetaflightController):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    device = torch.device(args_cli.device)
    num_envs = args_cli.num_envs
    rotor_dir = torch.tensor([1.0, -1.0, -1.0, 1.0], device=device).view(1, 4).repeat(num_envs, 1)
    # cache body id for wrench application
    body_id = robot.find_bodies("body")[0]

    sim_dt = sim.get_physics_dt()
    count = 0

    # Pre-allocate command tensor (A,E,T,R)
    command = torch.zeros(num_envs, 4, device=device)
    command[:, 2] = args_cli.hover_throttle  # throttle stick

    print(f"[INFO] Starting loop: hover_throttle={args_cli.hover_throttle} -> cmd_t={(args_cli.hover_throttle+1)/2:.3f}")

    while simulation_app.is_running():
        # Reset block
        if count % 2000 == 0:
            # reset counter inside block but keep global progression for sin() if desired
            # root state reset
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_vel = robot.data.default_joint_vel.clone()
            robot.set_joint_velocity_target(joint_vel)
            scene.reset()
            controller.reset(None)
            print("[INFO]: Resetting robot & controller state...")

        # Simple stick modulation example (optional yaw dither)
        # small yaw excitation to see response
        ##command[:, 3] = 0.1 * torch.sin(torch.tensor(count * sim_dt, device=device))
        controller.set_command(command)

        # Current body angular velocity (rad/s) from sim
        ang_vel = robot.data.root_ang_vel_b  # (num_envs,3)

        with torch.no_grad():
            des_ang_vel, motor_outputs = controller.compute(ang_vel)
            omega = motor_outputs * args_cli.omega_max  # (num_envs,4)
            # still compute joint velocities if needed for debugging (not applied)
            #joint_vel_target = omega * rotor_dir
            # Always compute thrust/torque for application
            thrust_torque_frd = controller.get_thrust_and_torque_command(
                omega_max=args_cli.omega_max,
                thrust_coeff=args_cli.thrust_coeff,
                drag_coeff=args_cli.drag_coeff,
                u=motor_outputs,
            )  # (num_envs,4): [T, Mx, My, Mz] in FRD
            # Split
            T_frd = thrust_torque_frd[:, 0]
            Mx_frd = thrust_torque_frd[:, 1]
            My_frd = thrust_torque_frd[:, 2]
            Mz_frd = thrust_torque_frd[:, 3]
            # Convert FRD -> FLU (x same, y negated, z negated). For thrust along +z_up we assume T is magnitude upward.
            # Force only along body +z (FLU) direction.
            force_flu = torch.zeros(num_envs, 3, device=device)
            force_flu[:, 2] = T_frd * args_cli.force_scale  # upward thrust
            # Moments: (Mx, My, Mz)_FLU = (Mx, -My, -Mz)_FRD
            moment_flu = torch.stack([
                Mx_frd * args_cli.force_scale,
                -My_frd * args_cli.force_scale,
                -Mz_frd * args_cli.force_scale,
            ], dim=1)
            # Reshape to (num_envs,1,3) as expected by API
            force_flu = force_flu.unsqueeze(1)
            moment_flu = moment_flu.unsqueeze(1)
            thrust_torque = thrust_torque_frd if args_cli.log_thrust_torque else None

        # Apply wrench to body
        try:
            robot.set_external_force_and_torque(force_flu, moment_flu, body_ids=body_id)
        except Exception as e:
            if count == 0:
                print(f"[WARN] Failed to apply external wrench: {e}")

        # Write data & step sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if count % args_cli.log_interval == 0:
            mean_u = motor_outputs.mean(dim=0).tolist()
            mean_omega = omega.mean(dim=0).tolist()
            des_rates = des_ang_vel[0].tolist()
            cur_rates = ang_vel[0].tolist()
            log_msg = (
                f"[LOG] step={count} cmd(A,E,T,R)={command[0].tolist()} des_rate(rad/s)={['%.2f'%r for r in des_rates]} "
                f"cur_rate={['%.2f'%r for r in cur_rates]} u={['%.2f'%x for x in mean_u]} omega(avg)={['%.0f'%w for w in mean_omega]}"
            )
            if args_cli.log_thrust_torque and thrust_torque is not None:
                tt0 = thrust_torque[0].tolist()
                log_msg += f" FRD[T,Mx,My,Mz]={['%.4e'%x for x in tt0]}"
                log_msg += f" AppliedForceZ={force_flu[0,0,2]:.4e} AppliedMoment={['%.4e'%m for m in moment_flu[0,0].tolist()]}"
            print(log_msg)

        count += 1


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 2.0, 4.0], [0.0, 0.0, 1.0])
    # Design scene
    scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Betaflight controller setup
    bf_params = BetaflightControllerParams(
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        dt=sim.get_physics_dt(),
    )
    controller = BetaflightController(bf_params)

    # Run the simulator
    run_simulator(sim, scene, controller)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()