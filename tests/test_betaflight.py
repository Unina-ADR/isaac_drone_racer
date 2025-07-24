# test_betaflight_controller.py

import torch
from controllers.betaflight import BetaflightController, BetaflightControllerParams
import sys

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Redirect stdout to a file to capture all print statements
    with open("betaflight_debug_output.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            # Set up parameters for a single environment
            params = BetaflightControllerParams(
                device=device,
                num_envs=1,
                dt=1/500
            )
            controller = BetaflightController(params)

            # Example command: [A, E, T, R] (roll, pitch, throttle, yaw) in [-1, 1]
            command = torch.tensor([[0.0, -0.3, -0.8, 0.0]], dtype=torch.float32)  # hover at mid throttle
            controller.set_command(command)

            # Example angular velocity: [roll_rate, pitch_rate, yaw_rate] in rad/s
            ang_vel = torch.zeros(1, 3, device=device, dtype=torch.float32)

            # Compute controller output
            des_ang_vel, u = controller.compute(ang_vel)

            print("Desired angular velocity (rad/s):", des_ang_vel)
            print("Motor outputs:", u)
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()