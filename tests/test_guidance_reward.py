import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- Setup ---
device = torch.device("cpu")
num_points = 50
y_vals = torch.linspace(-1.0, 1.0, num_points, device=device)
z_vals = torch.linspace(-1.0, 1.0, num_points, device=device)

# Waypoint at (1, 0, 0), yaw = 0
next_gate_pos = torch.tensor([1.0, 0.0, 0.0], device=device)
gate_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # yaw=0

# Guidance reward parameters
guidance_x_thresh = 3.0
guidance_tol = 0.2
k_rejection = 2.0
gate_height = 1.5
gate_width = 1.5

def quat_inv(q):
    # For yaw=0, this is identity
    return q

def quat_apply(q, v):
    # For yaw=0, this is identity
    return v

def compute_guidance_reward(
    drone_pos_gate_frame: torch.Tensor,
    gate_height: float,
    gate_width: float,
    guidance_x_thresh: float = 3.0,
    guidance_tol: float = 0.2,
    k_rejection: float = 2.0,
) -> torch.Tensor:
    x, y, z = (drone_pos_gate_frame[..., i] for i in range(3))
    layer_x = -torch.sgn(x) / guidance_x_thresh * x + 1
    layer_x.clamp_(min=0.0)
    guidance_x = -(layer_x ** 2)
    tol = torch.where(x > 0, 0.5, guidance_tol)
    yz_scale = (
        (1 - guidance_x) * tol * ((z ** 2 + y ** 2) / ((z / gate_height) ** 2 + (y / gate_width) ** 2)) ** 0.5
    )
    yz_scale.nan_to_num_(nan=1.0)
    guidance_yz = torch.where(
        x > 0,
        k_rejection * torch.exp(-0.5 * (y ** 2 + z ** 2) / yz_scale),
        (1 - torch.exp(-0.5 * (y ** 2 + z ** 2) / yz_scale)),
    )
    guidance = guidance_x + guidance_yz
    return guidance

# Slices to visualize
x_slices = [-1.0, 0.0, 0.5, 1.0, 2.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(x_slices)))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# To normalize color mapping across all slices, find global min/max
all_rewards = []
for x_slice in x_slices:
    Y, Z = torch.meshgrid(y_vals, z_vals, indexing='ij')
    X = torch.full_like(Y, x_slice)
    drone_pos = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    next_gate_pos_expanded = next_gate_pos.expand_as(drone_pos)
    gate_quat_expanded = gate_quat.expand(drone_pos.shape[0], 4)
    gate_quat_inverse = quat_inv(gate_quat_expanded)
    gate_to_drone = drone_pos - next_gate_pos_expanded
    drone_pos_gate_frame = quat_apply(gate_quat_inverse, gate_to_drone)
    reward = compute_guidance_reward(
        drone_pos_gate_frame,
        gate_height,
        gate_width,
        guidance_x_thresh,
        guidance_tol,
        k_rejection,
    ).reshape(num_points, num_points)
    all_rewards.append(reward)
all_rewards_tensor = torch.stack(all_rewards)
global_min = all_rewards_tensor.min().item()
global_max = all_rewards_tensor.max().item()

# Plot each slice
for idx, x_slice in enumerate(x_slices):
    Y, Z = torch.meshgrid(y_vals, z_vals, indexing='ij')
    X = torch.full_like(Y, x_slice)
    drone_pos = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    next_gate_pos_expanded = next_gate_pos.expand_as(drone_pos)
    gate_quat_expanded = gate_quat.expand(drone_pos.shape[0], 4)
    gate_quat_inverse = quat_inv(gate_quat_expanded)
    gate_to_drone = drone_pos - next_gate_pos_expanded
    drone_pos_gate_frame = quat_apply(gate_quat_inverse, gate_to_drone)
    reward = compute_guidance_reward(
        drone_pos_gate_frame,
        gate_height,
        gate_width,
        guidance_x_thresh,
        guidance_tol,
        k_rejection,
    ).reshape(num_points, num_points)

    normed_reward = (reward.cpu().numpy() - global_min) / (global_max - global_min + 1e-8)
    surf = ax.plot_surface(
        X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), 
        facecolors=plt.cm.viridis(normed_reward),
        rstride=1, cstride=1, antialiased=True, shade=False, alpha=0.7
    )

# Add colorbar for the reward
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(np.linspace(global_min, global_max, 100))
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Guidance Reward')

# Set labels and title
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Guidance Reward Slices at Different x Positions")
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()