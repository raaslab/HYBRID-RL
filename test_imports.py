# test_imports.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/env")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/cameras")

try:
    from env.mocks import MockGripper, MockRobot
    from cameras.camera import CameraDriver
    print("Imports successful!")
except ModuleNotFoundError as e:
    print(e)