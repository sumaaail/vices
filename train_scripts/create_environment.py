# creating the world
from robosuite.models import MujocoWorldBase
word = MujocoWorldBase()

# creating robot
from robosuite.models.robots import Sawyer
mujoco_robot = Sawyer()

# adding a gripper
from robosuite.models.grippers import gripper_factory
gripper = gripper_factory('TwoFingerGripper')
gripper.hide_visualization()
mujoco_robot.add_gripper("right_hand, gripper")

# add robot to the world
mujoco_robot.set_base_xpos([0,0,0])
word.merge(mujoco_robot)

# initialize TableArena instance
mujoco_arena = TableArena()