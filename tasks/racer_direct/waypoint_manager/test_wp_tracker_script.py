import torch
from track_generator import generate_track
from waypoint_tracker import WaypointTracker

# Define a simple track with two waypoints
def make_test_track():
    track_config = {
        "1": {"pos": (1.0, 0.0, 0.0), "yaw": 0.0},
        "2": {"pos": (3.0, 0.0, 0.0), "yaw": 0.0},
    }
    return generate_track(track_config)


def main():
    num_envs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track = make_test_track()
    # Patch track to have num_objects attribute if needed
    if not hasattr(track, "num_objects"):
        track.num_objects = len(track.rigid_objects)
    # Patch track.data to have object_pos_w and object_quat_w for test
    # For this test, we assume all waypoints are at the same positions for all envs
    wp_pos = torch.tensor([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # env 0
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # env 1
    ], dtype=torch.float32, device=device)
    wp_quat = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    ], dtype=torch.float32, device=device)
    class DummyData:
        pass
    track.data = DummyData()
    track.data.object_pos_w = wp_pos
    track.data.object_quat_w = wp_quat

    tracker = WaypointTracker(num_envs=num_envs, track=track)
    tracker.set_waypoint_data()

    # Test 1: Drone at (0.0, 0.0, 0.5) in both envs
    drone_state = torch.zeros(num_envs, 13, device=device)
    drone_state[:, :3] = torch.tensor([0.0, 0.0, 0.5], device=device)
    next_wp_id = torch.zeros(num_envs, dtype=torch.long, device=device)
    tracker.set_init_drone_state_next_wp(drone_state, next_wp_id)
    wp_passing, next_wp = tracker.compute(drone_state)
    print("Test 1: Drone at (0,0,0.5)")
    print("  Waypoint passing:", wp_passing.cpu().numpy())
    print("  Next waypoint id:", next_wp.cpu().numpy())

    # Test 2: Drone at (1.0, 0.0, 0.0) (midway between wp1 and wp2) in both envs
    drone_state[:, :3] = torch.tensor([1.0, 0.0, 0.0], device=device)
    tracker.set_init_drone_state_next_wp(drone_state, next_wp_id)
    wp_passing, next_wp = tracker.compute(drone_state)
    print("Test 2: Drone at (1,0,0)")
    print("  Waypoint passing:", wp_passing.cpu().numpy())
    print("  Next waypoint id:", next_wp.cpu().numpy())

if __name__ == "__main__":
    main()
