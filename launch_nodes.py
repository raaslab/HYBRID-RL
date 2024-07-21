from dataclasses import dataclass
from pathlib import Path

import tyro

from zmq_core.robot_node import ZMQServerRobot

import numpy as np

@dataclass
class Args:
    ###------ Hardware ------
    # hostname: str = "192.168.77.243"
    hostname: str = "10.104.56.158"
    robot_ip: str = "192.168.77.21"
    robot_port: int = 50003  # for trajectory
    robot: str = "ur"

def launch_robot_server(args: Args):
    port = args.robot_port

    if args.robot == "ur":
        from robots.ur import URRobot

        # robot = URRobot(robot_ip=args.robot_ip)
        robot = URRobot(robot_ip=args.robot_ip, no_gripper=False)

    else:
        raise NotImplementedError(
            f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
        )
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))