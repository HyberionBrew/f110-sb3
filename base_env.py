import gymnasium as gym
import f110_gym
#from gymnasium.wrappers import RescaleAction
from code.wrappers import F110_Wrapped, ThrottleMaxSpeedReward, FixSpeedControl, RandomStartPosition, FrameSkip
from code.wrappers import FlattenAction,SpinningReset,ProgressObservation, LidarOccupancyObservation, VelocityObservationSpace, NormalizeVelocityObservation
from code.wrappers import NormalizePose
from stable_baselines3.common.env_checker import check_env
from single_agent_env import ActionDictWrapper
from code.wrappers import ProgressReward
import numpy as np
from stable_baselines3.common import logger
# from f110_gym.envs.base_classes import Integrator
# from stable_baselines3.common.logger import Logger, get_logger

from gymnasium.spaces import Box
from typing import Union
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers import ClipAction
from code.reward import MixedReward
from gymnasium.wrappers import TimeLimit

class RescaleAction2(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleAction
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4")
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1,1,1]))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 0.75])
        >>> wrapped_env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.alltrue(obs == wrapped_env_obs)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray],
        max_action: Union[float, int, np.ndarray],
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        gym.utils.RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )
        gym.ActionWrapper.__init__(self, env)

        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )

        self.action_space = Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action

class MinChangeReward(gym.Wrapper):
    def __init__(self, env, collision_penalty: float = 10.0):
        assert collision_penalty >= 0.0, f"penalty must be >=0 and will be subtracted to the reward ({collision_penalty}"
        self._collision_penalty = collision_penalty
        super(MinChangeReward, self).__init__(env)
        print(self.action_space)
        self._last_action = np.zeros((1, self.action_space.shape[1]))
        print(self._last_action)
        self._last_action[0][:] = 1.0 # last velocity set to 1.0

    def normalize_action(self, action):
        low = self.env.action_space.low       
        high = self.env.action_space.high   
        normalized_action = 2 * (action - low) / (high - low) - 1
        return normalized_action


    def step(self, action):
        obs, _, done, truncated, info = super(MinChangeReward, self).step(action)
        if done:
            reward = - self._collision_penalty
        else:
            reward = 1 - 1/len(action) * np.linalg.norm(self.normalize_action(self._last_action) - 
                                        self.normalize_action(action))**2
        self._last_action = action

        return obs, reward, done, truncated, info

class MinActionReward(gym.Wrapper):
    def __init__(self, env, collision_penalty: float = 10.0 , mean_speed:float = 1.5):
        assert collision_penalty >= 0.0, f"penalty must be >=0 and will be subtracted to the reward ({collision_penalty}"
        self.collision_penalty = collision_penalty
        self.target_velocity = mean_speed
        super(MinActionReward, self).__init__(env)

    def normalize_action(self, action):
        low = self.env.action_space.low[0]
        high = self.env.action_space.high[0]
        steering_action = action[0][0]
        
        normalized_steering = 2 * (steering_action - low[0]) / (high[0] - low[0]) - 1
        if len(action[0]) == 2:
            velocity_action = action[0][1]
            half_range = np.minimum(self.target_velocity - low[1], high[1] - self.target_velocity)
            
            # Normalize the action around target_velocity
            normalize_velocity = (velocity_action - self.target_velocity) / half_range
            norm_action = np.array([normalized_steering, normalize_velocity])
        else:
            norm_action = np.array([normalized_steering])
        norm_action = np.clip(norm_action, -1, 1)
        return norm_action

    def step(self, action):
        obs, _, done, truncated, info = super(MinActionReward, self).step(action)
        #print(action)
        action = self.normalize_action(action) #[self._normalize_action(key, val) for key, val in action.items()]
        #print("normalized", action)
        assert np.all((abs(action) <= 1))
        if done:
            reward = - self.collision_penalty
        else:
            reward = 1 - (1 / len(action) * np.linalg.norm(action) ** 2)
        #print("reward", reward)
        return obs, reward, done, truncated, info

class MixedGymReward(gym.Wrapper):
    def __init__(self, env, **reward_config):
        # add checks if in vectorized env??
        super().__init__(env)
        self.reward = MixedReward(env, env.track, **reward_config)
        self.all_rewards = []
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        reward, rewards = self.reward(observation, action, observation['collisions'][0], done)
        # print(reward)
        # do it with logging frequency
        #logger_instance = get_logger()
        #logger.record("reward", reward)
        #logger.record("reward_TD", rewards[0])
        self.all_rewards = rewards
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        pose = (observation['poses_x'][0], observation['poses_y'][0])
        # print(pose)
        self.reward.reset(pose)
        return observation, info


class ReduceSpeedActionSpace(gym.ActionWrapper):
    def __init__(self,env, low, high):
        super(ReduceSpeedActionSpace, self).__init__(env)
        self.ac_low = self.action_space.low
        self.ac_high = self.action_space.high
        self.ac_low[0][1] = low
        self.ac_high[0][1] = high
        self.action_space = Box(low=self.ac_low, high=self.ac_high, shape=(1,2), dtype=np.float32)

    def action(self, action):
        # clip action according
        return action

rewards = {"TD": ProgressReward,
           "MinAct": MinActionReward,
           "MinChange": MinChangeReward}

standard_config = {
    "collision_penalty": -50.0,
    "progress_weight": 1.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0, # has bug
    "inital_velocity": 1.5,
    "normalize": False,
}


def make_base_env(map= "Infsaal", fixed_speed=None, 
                  random_start=True, reward = "TD", 
                  eval=False, reward_config = standard_config):
    
    env = gym.make("f110_gym:f110-v0",
                    config = dict(map=map,
                    num_agents=1),
                    render_mode="human_fast",
                    ) #integrator=Integrator.euler)
    #print(env.action_space)
    #print(env.min_action)
    # env.min_action = np.array([-1.0, -1.0])
    
    # print(new_env.action_space)
    # print(env.action_space)
    env = RandomStartPosition(env, increased=[140,170], likelihood = 0.2)
    # make a spin wrap detector
    # print("HI")
    env = ReduceSpeedActionSpace(env, 0.5, 2.0)
    # add a reset if the velocity x +y falls below a threshold
    env = ClipAction(env)
    # env = rewards[reward](env) #ProgressReward(env)
    env = SpinningReset(env, maxAngularVel= 5.0)
    env = MixedGymReward(env, **reward_config)
    # env = MinSpeedReset(env, min_speed=0.4)
    #env = ActionDictWrapper(env, fixed_speed=fixed_speed)
    if fixed_speed is not None:
        env = FixSpeedControl(env, fixed_speed=fixed_speed)
    
    env = ProgressObservation(env)
    env = LidarOccupancyObservation(env, resolution=0.25)
    # print(env.observation_space)
    env = gym.wrappers.FilterObservation(env, filter_keys=["lidar_occupancy","linear_vels_x", "linear_vels_y", "ang_vels_z", "poses_x", "poses_y", "progress"]) #, "angular_vels_z"])
    #TODO! add pose to observation space
    env = VelocityObservationSpace(env)
    env = NormalizeVelocityObservation(env)
    # env = AddPreviousAction(env)
    # env = gym.wrappers.FilterObservation(env, filter_keys=["lidar_occupancy","linear_vels_x", "linear_vels_y"])
    #env = PoseObservationSpace(env)
    env = NormalizePose(env)
    env = FrameSkip(env, skip=5) # make it 20 HZ from 100 HZ
    
    #print(env.observation_space)
    env = RescaleAction2(env, min_action=-1.0,max_action= 1.0)
    env = TimeLimit(env, max_episode_steps=500)
    return env

from stable_baselines3 import PPO
if __name__ == "__main__":
    print("Starting test ...")
    #env = make_base_env(fixed_speed=None)
    #check_env(env, warn=True, skip_render_check=False)
    env = make_base_env(fixed_speed=None)
    check_env(env, warn=True, skip_render_check=False)
    eval_env = make_base_env(fixed_speed=None,
                random_start =True,
                eval=True)
    eval_env = TimeLimit(eval_env, max_episode_steps=100)
    model = PPO("MultiInputPolicy", eval_env, verbose=1, device='cpu')
    model.learn(total_timesteps=500, progress_bar=True)
    print("Finished Test")