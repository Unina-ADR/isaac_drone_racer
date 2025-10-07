import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quat_from_euler_xyz(roll, pitch, yaw):
    """Convert euler angles to quaternion (w, x, y, z)."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=1)

def quat_apply(q, v):
    """Apply quaternion q to vector v."""
    # q: (N, 4), v: (N, 3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Quaternion multiplication: q * v * q^-1
    # Convert v to quaternion with w=0
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    qw = w
    qx = x
    qy = y
    qz = z
    # Calculate q * v
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    v_rot = torch.stack([
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx
    ], dim=1)
    return v_rot


# Dummy values for one environment
# Drone at origin, facing along x-axis (no rotation)
drone_pos = torch.tensor([[0.0, 0.0, 0.0]])
# Input drone_att as rpy angles (roll, pitch, yaw)
drone_rpy = torch.tensor([[0.1, 0.8, 0.0]])  # Example: roll=0.1, pitch=0.6, yaw=0.0
# Convert rpy to quaternion automatically
drone_att = quat_from_euler_xyz(drone_rpy[:, 0], drone_rpy[:, 1], drone_rpy[:, 2])

next_gate_pos = torch.tensor([[2.0, 1.0, 0.5]])

# Vector to gate
vec_to_gate = next_gate_pos - drone_pos
vec_to_gate = vec_to_gate / torch.norm(vec_to_gate, dim=1, keepdim=True)

# Drone x-axis
x_axis = torch.tensor([[1.0, 0.0, 0.0]])

# Camera rotation: 50 degrees around y-axis (in radians)
cam_roll = torch.tensor([[0.0]])
cam_pitch = torch.tensor([[-0,523599266]])  # -50 degrees in radians
cam_yaw = torch.tensor([[0.0]])


# Camera quaternion
cam_quat = quat_from_euler_xyz(cam_roll, cam_pitch, cam_yaw)
# Apply drone attitude to x_axis
# For identity quaternion, this is just x_axis
# If you want to test with a rotated drone, change drone_att

drone_x_axis = quat_apply(drone_att, x_axis)
drone_x_axis = drone_x_axis / torch.norm(drone_x_axis, dim=1, keepdim=True)
cam_x_axis = quat_apply(cam_quat, drone_x_axis)
cam_x_axis = cam_x_axis / torch.norm(cam_x_axis, dim=1, keepdim=True)

# Compute angle between cam_x_axis and vec_to_gate
cam_x = cam_x_axis[0].numpy().flatten()
gate_vec = vec_to_gate[0].numpy()
dot = np.dot(cam_x, gate_vec)
cosine = dot / (np.linalg.norm(cam_x) * np.linalg.norm(gate_vec))
angle = np.arccos(np.clip(cosine, -1.0, 1.0))
angle_deg = np.degrees(angle)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot drone reference frame at origin
# ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='Drone X')
# ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Drone Y')
# ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Drone Z')

# Plot cam_x_axis
ax.quiver(0, 0, 0, cam_x[0], cam_x[1], cam_x[2], color='orange', label='Camera X')

# Plot vector to gate
ax.quiver(0, 0, 0, gate_vec[0], gate_vec[1], gate_vec[2], color='purple', label='To Gate')

# Gate frame visualization
# Gate position and orientation (identity quaternion for now)
gate_pos = next_gate_pos[0].numpy()
gate_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

def quat_apply_np(q, v):
    """Apply quaternion q to vector v (numpy version)."""
    w, x, y, z = q
    # Convert v to quaternion with w=0
    vx, vy, vz = v
    # Quaternion multiplication: q * v * q^-1
    # For identity quaternion, returns v
    # For general quaternion, use rotation matrix
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R @ v

# Gate axes in local frame
gate_x = quat_apply_np(gate_quat, np.array([1, 0, 0]))
gate_y = quat_apply_np(gate_quat, np.array([0, 1, 0]))
gate_z = quat_apply_np(gate_quat, np.array([0, 0, 1]))

# Plot gate frame at gate position
ax.quiver(gate_pos[0], gate_pos[1], gate_pos[2], gate_x[0], gate_x[1], gate_x[2], color='magenta', label='Gate X')
ax.quiver(gate_pos[0], gate_pos[1], gate_pos[2], gate_y[0], gate_y[1], gate_y[2], color='cyan', label='Gate Y')
ax.quiver(gate_pos[0], gate_pos[1], gate_pos[2], gate_z[0], gate_z[1], gate_z[2], color='lime', label='Gate Z')

# Annotate angle
mid_vec = (cam_x + gate_vec) / 2
ax.text(mid_vec[0], mid_vec[1], mid_vec[2], f"Angle: {angle_deg:.2f}°", color='black')

# Plot world reference frame 1 meter below along z axis
ax.quiver(0, 0, -1, 1, 0, 0, color='darkred', label='World X')
ax.quiver(0, 0, -1, 0, 1, 0, color='darkgreen', label='World Y')
ax.quiver(0, 0, -1, 0, 0, 1, color='darkblue', label='World Z')

# Visualize drone attitude (rotated axes at origin)
def plot_drone_attitude(ax, quat, length=1.0):
    # Get rotation matrix from quaternion
    w, x, y, z = quat[0]
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    # Rotated axes
    x_axis = R @ np.array([1, 0, 0]) * length
    y_axis = R @ np.array([0, 1, 0]) * length
    z_axis = R @ np.array([0, 0, 1]) * length
    ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='salmon', label='Drone Att X')
    ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='lightgreen', label='Drone Att Y')
    ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='skyblue', label='Drone Att Z')

# Plot drone attitude
plot_drone_attitude(ax, drone_att.numpy())

ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.set_zlim([-1, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Drone Reference Frame, Camera X Axis, and Vector to Gate')
plt.show()
