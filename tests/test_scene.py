#!/usr/bin/env python3
"""
Visualizza la direzione dell'asse ottico della camera del drone
usando DroneBetaflightSceneCfg con IsaacLab.
"""

from isaaclab.app import AppLauncher

# 🔹 Inizializza IsaacLab (serve per caricare carb, omni, ecc.)
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Ora si possono importare i moduli di IsaacLab
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, matrix_from_quat
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

from tasks.drone_racer.drone_racer_env_cfg import DroneRacerSceneCfg


def main():
    sim_context = sim_utils.SimulationContext()
    
    # --- Crea la scena ---
    scene_cfg = DroneRacerSceneCfg(num_envs=1, env_spacing = 0.0)
    scene = InteractiveScene(scene_cfg)

    # Ottieni riferimenti
    robot = scene["robot"]
    cam_cfg = scene_cfg.tiled_camera.offset

    # Posizione e orientamento del drone
    root_state = robot.data.default_root_state
    drone_pos = root_state[:3]
    drone_quat = root_state[3:7]

    # Trasforma offset camera nel mondo
    cam_pos_w = drone_pos + quat_apply(drone_quat, torch.tensor(cam_cfg.pos))
    cam_quat_w = torch.tensor(cam_cfg.rot)

    # Asse ottico (z in ROS)
    rotmat = matrix_from_quat(cam_quat_w)
    forward_dir = rotmat[:, 2] * 0.3  # vettore direzione scalato

    print("🚁 Drone position:", drone_pos)
    print("🎥 Camera position:", cam_pos_w)
    print("➡️  Camera optical axis (world):", forward_dir)

    # --- Visualizza l’asse ottico come freccia ---
    arrow_cfg = VisualizationMarkersCfg(
        prim_path="/World/CameraAxis",
        markers={
            "arrow": dict(
                type="arrow",
                color=(1.0, 0.0, 0.0),  # rosso
                scale=(0.05, 0.05, 0.3),
            )
        },
    )
    arrow_marker = VisualizationMarkers(arrow_cfg)
    arrow_marker.visualize(
        positions=cam_pos_w.unsqueeze(0),
        orientations=cam_quat_w.unsqueeze(0),
    )

    # --- Avvia simulazione ---
    scene.reset_all()
    print("✅ Simulation started — chiudi la finestra per terminare.")

    while simulation_app.is_running():
        scene.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
