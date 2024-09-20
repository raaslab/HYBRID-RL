import numpy as np
import time
from urtde_controller2 import URTDEController, URTDEControllerConfig
from pynput import keyboard
from env.save_csv import save_frame
import datetime
from pathlib import Path

def save_start_status(status, filename="/home/amisha/Assistive_Feeding_Gello/start_status.txt"):
    with open(filename, "w") as f:
        f.write(status)



# Initial step size for each command
STEP_SIZE = 0.005
current_position = [0, 0, 0, 0, 0, 0, 0]  # Assuming initial position with 0 deltas and gripper open

def on_press(key):
    global current_position
    try:
        if key.char == 'w':
            print("w pressed")
            current_position[0] -= STEP_SIZE
        elif key.char == 's':
            print("s pressed")
            current_position[0] += STEP_SIZE
        elif key.char == 'a':
            print("a pressed")
            current_position[1] -= STEP_SIZE
        elif key.char == 'd':
            print("d pressed")
            current_position[1] += STEP_SIZE
        elif key.char == 'r':
            print("r pressed")
            current_position[-1] = 0.5
        elif key.char == 'e':
            print("e pressed")
            current_position[-1] = 0.0
        elif key.char == 'q':
            print("q pressed")
            current_position = [0, 0, 0, 0, 0, 0, current_position[-1]]
    except AttributeError:
        if key == keyboard.Key.up:
            print("up pressed")
            current_position[2] += STEP_SIZE
        elif key == keyboard.Key.down:
            print("down pressed")
            current_position[2] -= STEP_SIZE
        elif key == keyboard.Key.esc:
            save_start_status("Stopped")
            print("esc pressed")
            return False  # Stop listener

def teleoperation():
    cfg = URTDEControllerConfig(
        task="lift",
        controller_type="CARTESIAN_DELTA",
        max_delta=0.06,
        mock=0,
        hostname="10.104.56.158",
        robot_port=50003,
        robot_ip="192.168.77.21",
        hz=100
    )
    
    controller = URTDEController(cfg, task=cfg.task)
    time.sleep(5)
    
    # Keyboard Listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # save_start_status("Reached Start")

    # Set paths and file names
    csv_file_name = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    save_path = Path("data").expanduser()

    print("Teleoperation started. Use 'w', 'a', 's', 'd', 'up', 'down' keys to control the robot. Press 'esc' to stop.")
    
    while listener.running:
        try:
            controller.update(np.array(current_position))

            # Save robot state to CSV
            obs, _ = controller.get_state()
            joint_states = obs['joint_positions']
            ee_pos_quat = np.concatenate((obs['robot0_eef_pos'], obs['robot0_eef_quat']))
            dt = datetime.datetime.now()
            save_frame(save_path, dt, joint_states, ee_pos_quat, csv_file_name)


        except KeyboardInterrupt:
            save_start_status("Stopped")
            print("\nExiting...")

if __name__ == "__main__":
    teleoperation()
