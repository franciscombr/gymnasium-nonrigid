from os import path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}


class ShadowHandWireRelacateEnv(MujocoEnv, EzPickle):
    """
    
    ## Action Space
    The action space is a `Box(-1.0, 1.0, (30,), float32)`. The control actions are absolute angular positions of the hand joints. The input of the control actions is set to a range between -1 and 1 by scaling the real actuator angle ranges in radians.
    The elements of the action array are the following:
    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Linear translation of the full arm in x direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTx                           | slide | position (m)|
    | 1   | Linear translation of the full arm in y direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTy                           | slide | position (m)|
    | 2   | Linear translation of the full arm in z direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTz                           | slide | position (m)|
    | 3   | Angular up and down movement of the full arm                                            | -1          | 1           | -0.4 (rad)   | 0.25 (rad)  | A_ARRx                           | hinge | angle (rad) |
    | 4   | Angular left and right and down movement of the full arm                                | -1          | 1           | -0.3 (rad)   | 0.3 (rad)   | A_ARRy                           | hinge | angle (rad) |
    | 5   | Roll angular movement of the full arm                                                   | -1          | 1           | -1.0 (rad)   | 2.0 (rad)   | A_ARRz                           | hinge | angle (rad) |
    | 6   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.524 (rad) | 0.175 (rad) | A_WRJ1                           | hinge | angle (rad) |
    | 7   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.79 (rad)  | 0.61 (rad)  | A_WRJ0                           | hinge | angle (rad) |
    | 8   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_FFJ3                           | hinge | angle (rad) |
    | 9   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ2                           | hinge | angle (rad) |
    | 10  | Angular position of the DIP joint of the forefinger                                     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ0                           | hinge | angle (rad) |
    | 11  | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_MFJ3                           | hinge | angle (rad) |
    | 12  | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ2                           | hinge | angle (rad) |
    | 13  | Angular position of the DIP joint of the middle finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ0                           | hinge | angle (rad) |
    | 14  | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_RFJ3                           | hinge | angle (rad) |
    | 15  | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ2                           | hinge | angle (rad) |
    | 16  | Angular position of the DIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ0                           | hinge | angle (rad) |
    | 17  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.7(rad)    | A_LFJ4                           | hinge | angle (rad) |
    | 18  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_LFJ3                           | hinge | angle (rad) |
    | 19  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ2                           | hinge | angle (rad) |
    | 20  | Angular position of the DIP joint of the little finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ0                           | hinge | angle (rad) |
    | 21  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | A_THJ4                           | hinge | angle (rad) |
    | 22  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.3 (rad)   | A_THJ3                           | hinge | angle (rad) |
    | 23  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.26 (rad)  | 0.26(rad)   | A_THJ2                           | hinge | angle (rad) |
    | 24  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.52 (rad)  | 0.52 (rad)  | A_THJ1                           | hinge | angle (rad) |
    | 25  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | A_THJ0                           | hinge | angle (rad) |
    ## Observation Space
    The observation space is of the type `Box(-inf, inf, (39,), float64)`. It contains information about the angular position of the finger joints, the pose of the palm of the hand, as well as kinematic information about the ball and target.
    | Num | Observation                                                                 | Min    | Max    | Joint Name (in corresponding XML file) | Site/Body Name (in corresponding XML file) | Joint Type| Unit                     |
    |-----|-----------------------------------------------------------------------------|--------|--------|----------------------------------------|--------------------------------------------|-----------|------------------------- |
    | 0   | Translation of the arm in the x direction                                   | -Inf   | Inf    | ARTx                                   | -                                          | slide     | position (m)             |
    | 1   | Translation of the arm in the y direction                                   | -Inf   | Inf    | ARTy                                   | -                                          | slide     | position (m)             |
    | 2   | Translation of the arm in the z direction                                   | -Inf   | Inf    | ARTz                                   | -                                          | slide     | position (m)             |
    | 3   | Angular position of the vertical arm joint                                  | -Inf   | Inf    | ARRx                                   | -                                          | hinge     | angle (rad)              |
    | 4   | Angular position of the horizontal arm joint                                | -Inf   | Inf    | ARRy                                   | -                                          | hinge     | angle (rad)              |
    | 5   | Roll angular value of the arm                                               | -Inf   | Inf    | ARRz                                   | -                                          | hinge     | angle (rad)              |
    | 6   | Angular position of the horizontal wrist joint                              | -Inf   | Inf    | WRJ1                                   | -                                          | hinge     | angle (rad)              |
    | 7   | Angular position of the vertical wrist joint                                | -Inf   | Inf    | WRJ0                                   | -                                          | hinge     | angle (rad)              |
    | 8   | Horizontal angular position of the MCP joint of the forefinger              | -Inf   | Inf    | FFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 9   | Vertical angular position of the MCP joint of the forefinge                 | -Inf   | Inf    | FFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 10  | Angular position of the DIP joint of the forefinger                         | -Inf   | Inf    | FFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 11  | Horizontal angular position of the MCP joint of the middle finger           | -Inf   | Inf    | MFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 12  | Vertical angular position of the MCP joint of the middle finger             | -Inf   | Inf    | MFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 13  | Angular position of the DIP joint of the middle finger                      | -Inf   | Inf    | MFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 14  | Horizontal angular position of the MCP joint of the ring finger             | -Inf   | Inf    | RFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 15  | Vertical angular position of the MCP joint of the ring finger               | -Inf   | Inf    | RFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 16  | Angular position of the DIP joint of the ring finger                        | -Inf   | Inf    | RFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 17  | Angular position of the CMC joint of the little finger                      | -Inf   | Inf    | LFJ4                                   | -                                          | hinge     | angle (rad)              |
    | 18  | Horizontal angular position of the MCP joint of the little finger           | -Inf   | Inf    | LFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 19  | Vertical angular position of the MCP joint of the little finger             | -Inf   | Inf    | LFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 20  | Angular position of the DIP joint of the little finger                      | -Inf   | Inf    | LFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 21  | Horizontal angular position of the CMC joint of the thumb finger            | -Inf   | Inf    | THJ4                                   | -                                          | hinge     | angle (rad)              |
    | 22  | Vertical Angular position of the CMC joint of the thumb finger              | -Inf   | Inf    | THJ3                                   | -                                          | hinge     | angle (rad)              |
    | 23  | Horizontal angular position of the MCP joint of the thumb finger            | -Inf   | Inf    | THJ2                                   | -                                          | hinge     | angle (rad)              |
    | 24  | Vertical angular position of the MCP joint of the thumb finger              | -Inf   | Inf    | THJ1                                   | -                                          | hinge     | angle (rad)              |
    | 25  | Angular position of the IP joint of the thumb finger                        | -Inf   | Inf    | THJ0                                   | -                                          | hinge     | angle (rad)              |
    | 26  | x positional difference from the palm of the hand to initial point          | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 27  | y positional difference from the palm of the hand to initial point          | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 28  | z positional difference from the palm of the hand to initial point          | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 29  | x positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 30  | y positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 31  | z positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    
    ## Rewards
    The environment can be initialized in either a `dense` or `sparse` reward variant.
    In the `dense` reward setting, the environment returns a `dense` reward function that consists of the following parts:
    - `get_to_ball`: increasing negative reward the further away the palm of the hand is from the ball. This is computed as the 3 dimensional Euclidean distance between both body frames.
        This penalty is scaled by a factor of `0.1` in the final reward.
    - `ball_off_table`: add a positive reward of 1 if the ball is lifted from the table (`z` greater than `0.04` meters). If this condition is met two additional rewards are added:
        - `make_hand_go_to_target`: negative reward equal to the 3 dimensional Euclidean distance from the palm to the target ball position. This reward is scaled by a factor of `0.5`.
        -` make_ball_go_to_target`: negative reward equal to the 3 dimensional Euclidean distance from the ball to its target position. This reward is also scaled by a factor of `0.5`.
    - `ball_close_to_target`: bonus of `10` if the ball's Euclidean distance to its target is less than `0.1` meters. Bonus of `20` if the distance is less than `0.05` meters.
    The `sparse` reward variant of the environment can be initialized by calling `gym.make('AdroitHandReloateSparse-v1')`.
    In this variant, the environment returns the following `sparse` reward function that consists of the following parts:
    - `ball_close_to_target`: bonus of `10` if the ball's Euclidean distance to its target is less than `0.1` meters. Bonus of `20` if the distance is less than `0.05` meters.
    ## Starting State
    
    ## Episode End
    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 200 timesteps.
    
    ## Arguments
    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 400 make the environment as follows:
    ```python
    import gymnasium as gym
    env = gym.make('AdroitHandRelocate-v1', max_episode_steps=400) -> Not correct 
    ```
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        xml_file_path="/home/francisco/Desktop/DIGI2/gymnasium_tests/gymnasium-nonrigid/gymnasium_nonrigid/envs/assets/hand_e_plus/sr_hand_e_plus_wire_env.xml";
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(32,0), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self._model_names=MujocoModelNames(self.model)
        self.action_space=spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape)

        #Ids necessários para cálculo de rewards e assim
        #self.target_obj_site_id = self._model_names.site_name2id["target"]
        #self.S_grasp_site_id = self._model_names.site_name2id["S_grasp"]
        #self.obj_body_id = self._model_names.body_name2id["Object"]
        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng=0.5*(self.model.actuator_ctrlrange[:,1] - self.model.actuator_ctrlrange[:,0])

        EzPickle.__init__(self,**kwargs)
    
    def step(self,a):

        #advance simulation
        a=np.clip(a,-1.0,1.0)
        a = self.act_mean + a * self.act_rng  # mean center and scale

        self.do_simulation(a, self.frame_skip)
        obs = []

        #calculate reward
        reward = 0

        goal_achieved = True 
        #render visual window
        if self.render_mode == "human":
            self.render()

    

        return obs, reward, False, False, dict(success=goal_achieved)





