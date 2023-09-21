import gymnasium as gym
import f110_gym
#from gymnasium.wrappers import RescaleAction
from code.wrappers import F110_Wrapped, ThrottleMaxSpeedReward, FixSpeedControl, RandomStartPosition, FrameSkip
from code.wrappers import MaxLaps, FlattenAction,SpinningReset,ProgressObservation, LidarOccupancyObservation, VelocityObservationSpace
from code.wrappers import DownsampleLaserObservation, PoseStartPosition
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
        #print(action)
        #print(self.min_action)
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
        #print(self.action_space)
        self._last_action = np.zeros((1, self.action_space.shape[1]))
        #print(self._last_action)
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

import collections
class AugmentObservationsPreviousAction(gym.Wrapper):
    def __init__(self, env, inital_velocity=1.5):
        super().__init__(env)
        self.inital_velocity = inital_velocity
        obs_dict = collections.OrderedDict()
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        obs_dict['previous_action'] = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high,
                                                     dtype=self.action_space.dtype,
                                                     shape=self.action_space.shape)
        self.observation_space = gym.spaces.Dict(obs_dict)

        self.previous_action = np.zeros((self.action_space.shape), dtype=self.action_space.dtype)

        if len(self.previous_action[0]) == 2:
            self.previous_action[0][1] = inital_velocity

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs['previous_action'] = self.previous_action
        self.previous_action = action
        return obs, reward, done, truncated, info
    
    def reset(self, seed = None, options=None):
        #print("options 2")
        #print(options)
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous_action = np.zeros((self.action_space.shape), dtype=self.action_space.dtype)
        
        if len(self.previous_action[0]) == 2:
            self.previous_action[0][1] = options["velocity"] #self.inital_velocity
        obs['previous_action'] = self.previous_action
        return obs, info


class MixedGymReward(gym.Wrapper):
    def __init__(self, env, **reward_config):
        # add checks if in vectorized env??
        super().__init__(env)
        # super(MixedGymReward, self).__init__(env)
        self.reward = MixedReward(env, env.track, **reward_config)
        self.all_rewards = []
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        reward, rewards = self.reward(observation, action, 
                                      observation['collisions'][0], done)
        # print(reward)
        # do it with logging frequency
        #logger_instance = get_logger()
        #logger.record("reward", reward)
        #logger.record("reward_TD", rewards[0])
        self.all_rewards = rewards
        # add final reward to self.all_rewards, which is a dict
        self.all_rewards["final_reward"] = reward
        info["rewards"] = self.all_rewards
        info["collision"] = observation['collisions'][0]
        # print("reward",reward)
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        pose = (observation['poses_x'][0], observation['poses_y'][0])
        # print(pose)
        self.reward.reset(pose)
        return observation, info

class AppendObservationToInfo(gym.Wrapper):
    def __init__(self, env):
        super(AppendObservationToInfo, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Add observation to info
        info["observations"] = {}
        for key, value in obs.items():
            info["observations"][key] = value
        return obs, reward, done, truncated, info

class AppendActionToInfo(gym.Wrapper):
    def __init__(self, env, name = "action_delta"):
        super(AppendActionToInfo, self).__init__(env)
        self.name = name
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Add observation to info
        info[self.name] = action
        return obs, reward, done, truncated, info

def normalize(value, low, high):
    """Normalize value between -1 and 1."""
    return 2 * ((value - low) / (high - low)) - 1

def clip(value, low, high):
    """Clip a value between low and high."""
    return np.clip(value, low, high)

class NormalizeVelocityObservation(gym.ObservationWrapper):
    def __init__(self, env, low_vel=-20, high_vel=20):
        super().__init__(env)
        self.bounds = {
            'linear_vels_x': (low_vel, high_vel),
            'linear_vels_y': (low_vel, high_vel),
            'ang_vels_z': (low_vel, high_vel)
        }
        for key, (_, _) in self.bounds.items():
            self.observation_space[key] = Box(shape=(1,), low=-1.0, high=1.0)

    def observation(self, obs):
        for key, (low, high) in self.bounds.items():
            obs[key] = clip(obs[key], low, high)
            obs[key] = normalize(obs[key], low, high)
        return obs

class NormalizePose(gym.ObservationWrapper):
    def __init__(self, env, low=-30, high=30, theta_low= -np.pi*2, theta_high=np.pi*2):
        super().__init__(env)
        self.pose_bounds = {
            'poses_x': (low, high),
            'poses_y': (low, high),
            'poses_theta': (theta_low, theta_high)
        }
        for key, (low, high) in self.pose_bounds.items():
            self.observation_space[key] = Box(shape=(1,), low=-1.0, high=1.0)

    def observation(self, observation):
        for key, (low, high) in self.pose_bounds.items():
            observation[key] = clip(observation[key], low, high)
            observation[key] = normalize(observation[key], low, high)
        return observation

class NormalizePreviousAction(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.action_space.shape #TODO!
        #self.min = min
        #self.max = max
        #for key, (low, high) in self.pose_bounds.items():
        self.observation_space["previous_action"] = Box(shape=(1,2), low=-1.0, high=1.0)

    def observation(self, obs):
        low = self.env.action_space.low
        high = self.env.action_space.high
        # print(high)
        #print(low, high)
        #print("obs", obs["previous_action"])
        
        
        obs["previous_action"] = np.clip(obs["previous_action"], low, high)
        obs["previous_action"] = normalize(obs["previous_action"], low, high)
        #print("obs", obs["previous_action"])
        # print("obs", obs)
        return obs
    
class RandomResetVelocity(gym.Wrapper):
    def __init__(self, env, min_vel=0.0, max_vel=2.0, eval=False):
        super(RandomResetVelocity, self).__init__(env)
        self.min_vel = min_vel
        self.max_vel = max_vel 
        self.eval = eval
    def reset(self, seed= None, options=None):
        if options is None:
            options = {}
        
        if "velocity" not in options:
            # Randomly sample velocity between min_vel and max_vel
            random_velocity = np.random.uniform(self.min_vel, self.max_vel)
            options['velocity'] = random_velocity
        if self.eval:
            options['velocity'] = 0.0
        return super().reset(seed=seed, options=options)

import warnings
class AccelerationAndDeltaSteeringWrapper(gym.ActionWrapper):
    def __init__(self, env, max_acceleration, max_delta_steering, min_velocity, max_velocity, min_steering, max_steering, inital_velocity=0.5):
        super(AccelerationAndDeltaSteeringWrapper, self).__init__(env)

        self.action_space = gym.spaces.Box(low=np.array([[-max_delta_steering,-max_acceleration]]),
                                           high=np.array([[max_delta_steering,max_acceleration]]),
                                           shape = (1,2),
                                           dtype=np.float32)

        # For maintaining the current state of the agent
        self.inital_velocity = inital_velocity
        self.current_velocity = self.inital_velocity
        self.current_steering = 0.0

        # Define the bounds
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_steering = min_steering
        self.max_steering = max_steering

    def reset(self, seed=None, options=None):
        # Reset the current state when the environment is reset
        #print(options)
        #print("------")
        if options is None:
            options = {} 
        
        if "poses" in options: 
            warnings.warn("Only velocity will be set")   
        if "velocity" not in options:
            warnings.warn("Starting with velocity 0")
            options["velocity"] = 0.0

        self.current_velocity = options["velocity"]
        self.current_steering = 0.0
        return super().reset( seed=None, options=options)

    def action(self, action):
        # Extract acceleration and delta steering from the action
        delta_steering, acceleration = action[0]

        # Compute the new absolute values
        self.current_velocity += acceleration
        self.current_steering += delta_steering

        # Ensure the absolute values are within safe bounds
        self.current_velocity = np.clip(self.current_velocity, self.min_velocity, self.max_velocity)
        self.current_steering = np.clip(self.current_steering, self.min_steering, self.max_steering)
        # print(self.current_steering)
        # print(self.current_velocity)
        # Return the transformed action (absolute values)
        return np.asarray([[self.current_steering, self.current_velocity]])

from f110_orl_dataset import normalize_dataset

class NormalizeObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.normalizer = normalize_dataset.Normalize()
        lidar_space = self.observation_space["lidar_occupancy"]
        self.observation_space = self.normalizer.new_state_space
        self.observation_space["lidar_occupancy"] = lidar_space
    def observation(self, obs):
        #print("og")
        #print(obs)
        obs = self.normalizer.normalize_obs_batch(obs)
        #print(obs)
        #print("new")
        #print(obs)
        return obs



class ReduceSpeedActionSpace(gym.ActionWrapper):
    def __init__(self,env, low, high):
        super(ReduceSpeedActionSpace, self).__init__(env)
        #self.ac_low = self.action_space.low
        #self.ac_high = self.action_space.high
        #self.ac_low[0][1] = low
        #self.ac_high[0][1] = high
        self.action_space.low[0][1] = low #= Box(low=self.ac_low, high=self.ac_high, shape=(1,2), dtype=np.float32)
        self.action_space.high[0][1] = high
    def action(self, action):
        # clip action according
        return action

rewards = {"TD": ProgressReward,
           "MinAct": MinActionReward,
           "MinChange": MinChangeReward}

standard_config = {
    "collision_penalty": -50.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 1.0,
    "inital_velocity": 1.5,
    "normalize": False,
}

from config import VEL_LOW, VEL_HIGH, MAX_VEL, MIN_VEL, \
                    POSE_LOW, POSE_HIGH, SUBSAMPLE, POSE_THETA_LOW, POSE_THETA_HIGH

def make_base_env(map= "Infsaal", fixed_speed=None, 
                  random_start=True,
                  train_random_start = True,
                  eval=False, use_org_reward=False, 
                  pose_start = False,
                  reward_config = standard_config, acceleration =False):
    """
    Setup and return the base environment with required wrappers.
    """
    env = gym.make("f110_gym:f110-v0",
                    config = dict(map=map,
                    num_agents=1, 
                    params=dict(vmin=0.5, vmax=2.0)),
                    render_mode="human")

    # Random start wrapper
    if random_start:
        if train_random_start:
            # we are oversampling some harder start positions
            env = RandomStartPosition(env, increased=[140,170], likelihood = 0.2)
        else:
            env = RandomStartPosition(env)
    else:
        # in this case we always start at 0 (first point in raceline)
        if pose_start:
            env = PoseStartPosition(env)
        else:
            env = RandomStartPosition(env, increased=[0,1], likelihood = 1.0)

    # Clip the velocity
    # env = ReduceSpeedActionSpace(env, 0.5, 1.8)
    env = ReduceSpeedActionSpace(env, MIN_VEL, MAX_VEL)
    env = ClipAction(env)

    # restart if start spinning
    env = SpinningReset(env, maxAngularVel= 5.0)

    # only steering is learnable
    if fixed_speed is not None:
        env = FixSpeedControl(env, fixed_speed=fixed_speed)
    # print(env.action_space.low[0])
    
    # make it 20 HZ from 100 HZ
    env = FrameSkip(env, skip=5) 
    env = MixedGymReward(env, **reward_config)
    # print(env.action_space)
    env = AppendActionToInfo(env, name="action_raw")
    env = AugmentObservationsPreviousAction(env, inital_velocity=2.0)
    
    # convert to delta steering and acceleration

    # special wrapper for evaluation
    if eval:
        env = MaxLaps(env, max_laps=1, finished_reward=20, max_lap_time=1000, use_org_reward=use_org_reward)
    
    # add and filter observations
    env = ProgressObservation(env)
    env = DownsampleLaserObservation(env, subsample=SUBSAMPLE)
    env = gym.wrappers.FilterObservation(env, filter_keys=["lidar_occupancy","linear_vels_x", 
                                                           "linear_vels_y", "ang_vels_z", "previous_action", # ])
                                                           "poses_x", "poses_y", "poses_theta", 
                                                           "progress_sin", "progress_cos"]) #, "angular_vels_z"])
    


    # append all observations to the info dict
    env = AppendObservationToInfo(env)
    

    # Normalize the whole observation space
    env = NormalizeObservations(env)
    #env = NormalizePreviousAction(env)
    #env = NormalizeVelocityObservation(env, low_vel=VEL_LOW, high_vel=VEL_HIGH)
    #env = NormalizePose(env, low=POSE_LOW, high=POSE_HIGH, 
    #                   theta_low=POSE_THETA_LOW, theta_high=POSE_THETA_HIGH)
    # Change action space
    env = AccelerationAndDeltaSteeringWrapper(env, max_acceleration=0.05, max_delta_steering=0.05,
                                            min_velocity=env.action_space.low[0][1], max_velocity=env.action_space.high[0][1],
                                            min_steering=env.action_space.low[0][0], max_steering=env.action_space.high[0][0],
                                            inital_velocity=2.0)
    env = AppendActionToInfo(env, name="action_delta")
    env = RescaleAction2(env, min_action=-1.0,max_action=1.0)
    
    env = ClipAction(env)
    # add a timelimit TODO! maybe not include here
    env = TimeLimit(env, max_episode_steps=1000)
    env = RandomResetVelocity(env, min_vel=0.8, max_vel=1.5, eval=False)
    return env

from stable_baselines3 import PPO
from functools import partial
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    print("Starting test ...")
    #env = make_base_env(fixed_speed=None)
    #check_env(env, warn=True, skip_render_check=False)
    env = make_base_env(fixed_speed=None)
    check_env(env, warn=True, skip_render_check=False)
    eval_env = make_base_env(fixed_speed=None,
                random_start =True,
                eval=True)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)
    model = PPO("MultiInputPolicy", eval_env, verbose=1, device='cpu')
    model.learn(total_timesteps=5, progress_bar=True)
    
    partial_make_base_env = partial(make_base_env, 
                                fixed_speed=None,
                                random_start =True,)
    train_envs = make_vec_env(partial_make_base_env,
                n_envs=2,
                seed=np.random.randint(pow(2, 31) - 1),
                vec_env_cls=SubprocVecEnv)
    model = PPO("MultiInputPolicy", train_envs, verbose=1, device='cpu')
    
    print("Finished Test")