import os
import sys

# Add project root to PYTHONPATH if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from tasks.drone_racer.drone_racer_env_cfg import DroneRacerSceneCfg
import isaaclab.sim as sim_utils

def main():
    # Create simulation context (headless=False for viewer)
    sim = sim_utils.SimulationContext(headless=False)
    
    # Instantiate the scene configuration (single environment for visualization)
    scene_cfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    scene = scene_cfg.instantiate(sim)
    
    # Reset and render the scene
    sim.reset()
    sim.render()
    
    print("Scene loaded. Close the viewer window or press Ctrl+C to exit.")
    try:
        while sim.is_running():
            sim.step()
            sim.render()
    except KeyboardInterrupt:
        print("Exiting.")

if __name__ == "__main__":
    main()