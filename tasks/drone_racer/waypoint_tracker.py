import torch
from typing import Tuple


import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

#from track_generator import generate_track




class WaypointTracker:
    def __init__(self, num_envs, track: RigidObjectCollectionCfg):
        self.device = "cuda"
        self.num_envs = num_envs
        self.all_env_ids = torch.arange(num_envs, device=self.device)
        self.num_waypoints = track.num_objects
        self.track = track
        self.all_wp_id_un_sq = torch.arange(track.num_objects, device=self.device).unsqueeze(0)
        
        self.wp_pos = torch.zeros(self.num_envs, self.num_waypoints, 3, device=self.device)
        self.wp_x_axis = torch.zeros(self.num_envs, self.num_waypoints, 3, device=self.device)
        self.wp_y_axis = torch.zeros(self.num_envs, self.num_waypoints, 3, device=self.device)
        self.wp_z_axis = torch.zeros(self.num_envs, self.num_waypoints, 3, device=self.device)
        self.wp_y_dim = 0.75*torch.ones(self.num_envs, self.num_waypoints, device=self.device)
        self.wp_z_dim = 0.75*torch.ones(self.num_envs, self.num_waypoints, device=self.device)     
        
        self.is_wp_passed = torch.zeros(self.num_envs, self.num_waypoints, device=self.device, dtype=torch.bool)

        self.last_drone_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)

    def set_waypoint_data(self):

        self.wp_pos[:] = self.track.data.object_pos_w[self.all_env_ids, self.all_wp_id_un_sq, :]
        wp_quat = self.track.data.object_quat_w[self.all_env_ids, self.all_wp_id_un_sq, :]
        wp_mat = math_utils.quat_to_mat(wp_quat)

        self.wp_x_axis[:] = wp_mat[:, :, 0, :]
        self.wp_y_axis[:] = wp_mat[:, :, 1, :]
        self.wp_z_axis[:] = wp_mat[:, :, 2, :]

    def set_init_drone_state_next_wp(self, 
        drone_state: torch.Tensor, 
        next_wp_id: torch.Tensor, 
        env_id: torch.Tensor = None,
        ):
        if env_id is None:
            env_id = self.all_env_ids

        self.last_drone_pos[env_id] = drone_state[env_id, :3].unsqueeze(1)

        wp_passed = self.all_wp_id_un_sq < next_wp_id[env_id].unsqueeze(1)
        self.is_wp_passed[env_id] = wp_passed
        

    def compute(self, drone_state: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        """ ERCBUNNY CODE
        Checks waypoint passing and computes the next target waypoint id for all envs,
        based on the updated drone state and last drone state stored internally.
        This function is called during rollout.
        Make sure to call ''set_waypoint_data'' and ''set_init_drone_pos'' properly before using this function.
        
        Args:
        drone_state: full drone state tensor in (num_envs, 13) [pos, quat, lin_vel, ang_vel]
        Returns:
        waypoint passing flag
        next target waypoint id
        """
        self.last_drone_pos[:], self.is_wp_passed[:], wp_passing, next_wp_id = (
            _compute_script(
                wp_pos=self.wp_pos,
                wp_x_axis=self.wp_x_axis,
                wp_y_axis=self.wp_y_axis,
                wp_z_axis=self.wp_z_axis,
                wp_width=self.wp_width,
                wp_height=self.wp_height,
                is_wp_passed=self.is_wp_passed,
                last_drone_pos=self.last_drone_pos,
                drone_state=drone_state,
            )
        )

        return wp_passing, next_wp_id
    

@torch.jit.script
def _compute_script(
    wp_pos: torch.Tensor,
    wp_x_axis: torch.Tensor,
    wp_y_axis: torch.Tensor,
    wp_z_axis: torch.Tensor,
    wp_y_dim: torch.Tensor,
    wp_z_dim: torch.Tensor,
    is_wp_passed: torch.Tensor,
    last_drone_pos: torch.Tensor,
    drone_state: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    drone_pos = drone_state[:, :3].unsqueeze(1)  # [num_envs, 1, 3]
    drone_pos_diff = drone_pos - last_drone_pos  # [num_envs, 1, 3]
    last_drone_pos_to_wp = wp_pos - last_drone_pos  # [num_envs, num_waypoints, 3]

    #compute intersection point param (num_envs, num_waypoints, 1)
    # calculate intersection point param (num_envs, num_waypoints, 1)
    intersect_t_num = torch.sum(last_drone_pos_to_wp * wp_x_axis, dim=-1, keepdim=True)
    intersect_t_den = torch.sum(drone_pos_diff * wp_x_axis, dim=-1, keepdim=True)
    intersect_t = intersect_t_num / intersect_t_den

    # intersection point positions (num_envs, num_waypoints, 3)
    intersect_p = last_drone_pos + intersect_t * drone_pos_diff

    # vector from waypoint center to intersection point (num_envs, num_waypoints, 3)
    wp_to_intersect = intersect_p - wp_pos

    # project wp to intersect to y and z axes (num_envs, num_waypoints)
    intersect_proj_y = torch.sum(wp_to_intersect * wp_y_axis, dim=-1)
    intersect_proj_z = torch.sum(wp_to_intersect * wp_z_axis, dim=-1)

    # waypoint passing conditions (num_envs, num_waypoints)
    cond_dir = intersect_t_den.squeeze() > 0

    intersect_t_sq = intersect_t.squeeze()
    cont_t_nan = ~torch.isnan(intersect_t_sq)
    cond_t_lb = intersect_t_sq >= 0
    cond_t_ub = intersect_t_sq < 1

    cond_y_dim = intersect_proj_y.abs() < wp_y_dim / 2
    cond_z_dim = intersect_proj_z.abs() < wp_z_dim / 2

    cond_previous = is_wp_passed.roll(1, dims=1)
    cond_previous[:, 0] = True

    is_wp_passed_new = is_wp_passed | (
        cond_dir
        & cont_t_nan
        & cond_t_lb
        & cond_t_ub
        & cond_y_dim
        & cond_z_dim
        & cond_previous
    )

    # calculate wp passing indicator
    wp_passing = (is_wp_passed != is_wp_passed_new).any(dim=1)

    # calculate next waypoint id (num_envs, )
    next_wp_id = torch.eq(torch.cumsum(~is_wp_passed_new, dim=1), 1).max(dim=1).indices

    return drone_pos, is_wp_passed_new, wp_passing, next_wp_id