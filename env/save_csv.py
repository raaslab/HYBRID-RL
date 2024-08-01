import datetime
import csv
from pathlib import Path
import numpy as np
import os
import pyrealsense2 as rs
import cv2

def save_frame(folder: Path, timestamp: datetime.datetime, joint_positions, ee_pos_quat, file_name: str) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    recorded_file = os.path.join(folder, file_name)
    file_exists = os.path.isfile(recorded_file)
    
    try:
        # Save robot state to CSV
        # obs = env.get_obs()
        # joint_positions = obs["joint_positions"]
        # ee_pos_quat = obs["ee_pos_quat"][:-1]
        obs_combined = np.concatenate((joint_positions, ee_pos_quat))
        
        # print(recorded_file)
        with open(recorded_file, "a", newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['timestamp', 'shoulder_pan_angle', 'shoulder_lift_angle', 'elbow_angle', 'wrist1_angle', 'wrist2_angle', 'wrist3_angle', 'gripper_position', 'end_eff_x', 'end_eff_y', 'end_eff_z', 'end_eff_roll', 'end_eff_pitch', 'end_eff_yaw', 'gripper_position'])
            
            # Convert timestamp to time-only string format (HH:MM:SS.mmm)
            timestamp_str = timestamp.strftime('%H:%M:%S.%f')[:-3]  # [:-3] to truncate microseconds to milliseconds
            row_data = [timestamp_str] + list(obs_combined)
            writer.writerow(row_data)
            
        # print(f"Frame data appended to {recorded_file}")
    except Exception as e:
        print(f"Error saving robot state to CSV: {e}")
