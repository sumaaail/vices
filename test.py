import mujoco_py
import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np
from mujoco_py import GlfwContext

GlfwContext(offscreen=True)  # Trying to initialize an OpenGL context

env = HalfCheetahEnv()
test = env.reset()
mode_human = False  # Work well in human mode, don't work in "rgb_array" mode
image = env.render(mode="human" if mode_human else "rgb_array")
print("image is: ")
print(image)