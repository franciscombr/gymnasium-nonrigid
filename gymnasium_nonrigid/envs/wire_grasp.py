import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand import (
    MujocoManipulateEnv,
    MujocoPyManipulateEnv
)

MANIPULATE_BLOCK_XML = "/home/francisco/Desktop/DIGI2/gymnasium_tests/gymnasium-nonrigid/gymnasium_nonrigid/envs/assets/hand/manipulate_block_touch_sensors.xml"

class MujocoHandWireEnv(MujocoManipulateEnv,EzPickle):
    