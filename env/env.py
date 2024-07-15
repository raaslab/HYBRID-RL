import time
from typing import Any, Dict, Optional

import numpy as np

from cameras.camera import CameraDriver
from robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        # print("Commanding joint state")
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def update_desired_ee_pose(self, pose: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        self._robot.command_eef_pose(pose)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        return observations

    def get_ee_pose(self) -> np.ndarray:
        """
        Gives out the end effector position of the leader robot.
        """
        robot_end_eff_pos = self._robot.get_observations()["ee_pos_quat"]
        
        return robot_end_eff_pos

    def set_home_pose(self, home_pose: np.ndarray) -> None:
        """Sets the home pose for `go_home()` to use."""
        self.home_pose = home_pose

    def go_home(self, blocking=True) -> None:
        """Calls move_to_joint_positions to the current home positions."""
        if blocking is False:
            assert (
                self.home_pose is not None
            ), "Home pose not assigned! Call 'set_home_pose(<joint_angles>)' to enable homing"
            return self.move_to_joint_positions(
                positions=self.home_pose, delta=False)

    def move_to_joint_positions(self, positions: np.ndarray, delta: bool = False):
        """Moves the robot to the specified joint positions.

        Args:
            positions (torch.Tensor): The joint positions to move the robot to.
            delta (bool, optional): Whether the positions are relative to the current positions. Defaults to False.

        Returns:
            bool: True if the robot successfully moved to the specified positions, False otherwise.
        """
        print("move_to_joint_positions")
        curr_joints = self._robot.get_observations()["joint_positions"]
        # print("curr_joints", curr_joints)
        # print("len(list(positions))", len(list(positions)), "len(curr_joints)", len(curr_joints))
        if len(list(positions)) == len(curr_joints):
            max_delta = (np.abs(curr_joints - positions)).max()
            print("max_delta", max_delta)   
            steps = min(int(max_delta / 0.01), 100)
            print("steps", steps)
        for jnt in np.linspace(curr_joints, positions, steps):
            self.step(jnt)
        return True
         
    def move_to_eef_positions(self, positions: np.ndarray, delta: bool = False):
        """Moves the robot to the specified joint positions.

        Args:
            positions (torch.Tensor): The joint positions to move the robot to.
            delta (bool, optional): Whether the positions are relative to the current positions. Defaults to False.

        Returns:
            bool: True if the robot successfully moved to the specified positions, False otherwise.
        """
        print("move_to_eef_positions")
        curr_pose = self._robot.get_observations()["ee_pos_quat"]
        gripper_pos = self._robot.get_observations()["gripper_position"]

        curr_pose = np.concatenate([curr_pose, gripper_pos])
        # print("curr_joints", curr_joints)
        print("len(list(positions))", list(positions), "len(curr_joints)", curr_pose)
        print("len(reset_pose)", len(positions), "len(curr_pose)", len(curr_pose))
        if len(positions) == len(curr_pose):
            rpy_delta = np.abs(curr_pose[3:6] - positions[3:6])   
            for angle in rpy_delta:
                if abs(angle) > 0.75:
                    raise ValueError("Angle difference is too large")
                
            max_delta = (np.abs(curr_pose - np.array(positions))).max()
            print("max_delta", max_delta)   
            steps = min(int(max_delta / 0.001), 100)

            for pose in np.linspace(curr_pose, positions, steps):
                print("pose", pose)
                self.update_desired_ee_pose(pose)
                time.sleep(0.001)

        return True
            

def main() -> None:
    pass


if __name__ == "__main__":
    main()


