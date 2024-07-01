import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import h5py


# Load waypoints from hdf5 file
hdf5_path = '/home/amisha/ibrl/release/data/robomimic/square/processed_data96.hdf5'
waypoints = []
gripper_controls = []

with h5py.File(hdf5_path, 'r') as f:
    actions = f["data/demo_0/actions"][:]



def main():
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=controller_config
    )
    obs = env.reset()
    # max_steps = 100  # Example of a maximum number of steps
    for action in actions:
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Task completed.")
            break
if __name__ == "__main__":
    main()