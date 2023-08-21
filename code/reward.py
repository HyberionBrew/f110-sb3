import numpy as np
from f110_gym.envs.track import Track
from code.wrappers import Progress
from typing import Tuple
import gymnasium as gym

from gymnasium.wrappers.normalize import RunningMeanStd, update_mean_var_count_from_moments

# adapted slightly from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
class NormalizeReward():
        def __init__(self,
                gamma: float = 0.99,
                epsilon: float = 1e-8,):
            self.return_rms = RunningMeanStd(shape=())
            self.gamma = gamma
            self.epsilon = epsilon
            self.returns = np.zeros(1)
        def step(self,rews , done):
            self.returns = self.returns * self.gamma * (1 - done) + rews 
            rews = self.normalize(rews)
            return rews
        def normalize(self, rews):
            self.return_rms.update(self.returns)
            return rews / np.sqrt(self.return_rms.var + self.epsilon)
        
class ProgressReward(object):
    def __init__(self, track:Track) -> None:
        self._current_progress = None
        self.progress_tracker = Progress(track)
    
    def reset(self, new_pose: Tuple[float, float]):
        pose = np.array(new_pose)
        # print(pose)
        pose = pose[np.newaxis, :]
        self.current_progress = self.progress_tracker.get_progress(pose)[0]

    def __call__(self, pose: Tuple[float, float]):
        pose = np.array(pose)
        pose = pose[np.newaxis, :]
        new_progress = self.progress_tracker.get_progress(pose)[0]
        delta_progress = 0
        # in this case we just crossed the finishing line!
        if (new_progress - self.current_progress) < -0.5:
            delta_progress = (new_progress + 1) - self.current_progress
        else:
            delta_progress = new_progress - self.current_progress
        delta_progress = max(delta_progress, 0)
        delta_progress *= 100
        self.current_progress = new_progress
        reward = delta_progress
        return reward

"""
@brief     Reward function based on the distance from the raceline
"""
class RacelineDeltaReward(object):
    def __init__(self, track: Track) -> None:
        xs = track.raceline.xs
        ys = track.raceline.ys
        self.centerline = np.stack((xs, ys), axis=-1)

    def __call__(self, pose: Tuple[float, float]) -> float:
        # calculate the squared distances to all points on the raceline
        distances = np.sum((self.centerline - np.array(pose))**2, axis=1)
        # get the minimum squared distance
        min_distance_squared = np.min(distances)
        # take the square root to get the actual minimum distance
        min_distance = np.sqrt(min_distance_squared)
        # the negative of the distance, meaning closer to the line is better
        return -min_distance
"""
@brief    Reward function based on the change in steering angle
"""
class MinSteeringChangeReward(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self._last_action = 0.0
    def __call__(self, action: np.ndarray) -> float:
        # penalize the change in steering angle
        assert(len(action) == 1 or (len(action) == 2))
        
        delta = (action[0] - self._last_action) **2
        delta = np.clip(delta, self.low, self.high)
        reward = -delta
        self._last_action = action[0]
        return reward
    def reset(self):
        self._last_action = 0.0

"""
@brief    Reward function based on the change in velocity
"""
class MinVelocityChangeReward(object):
    def __init__(self, low, high, inital_velocity=1.5):
        self.low = low
        self.high = high
        self.inital_velocity = inital_velocity
        self._last_velocity = inital_velocity
    def __call__(self, vel_x, vel_y) -> float:
        # penalize the change in velocity angle
        velocity = np.sqrt(vel_x**2 + vel_y**2)
        delta = (velocity- self._last_velocity) **2
        delta = np.clip(delta, self.low, self.high)
        reward = -delta
        self._last_velocity = velocity
        return reward
    def reset(self):
        self._last_velocity = self.inital_velocity

"""
@brief    Reward function based on the speed of the car
"""
class VelocityReward(object):
    def __init__(self, track: Track) -> None:
        pass

    def __call__(self, velocity_x: float, velocity_y) -> float:
        # faster is better
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)
        return velocity



class MixedReward(object):
    def __init__(self, env: gym.Env,
                 track: Track, 
                 collision_penalty: float = -10.0, 
                 progress_weight: float = 1.0, 
                 raceline_delta_weight: float = 0.0, 
                 velocity_weight: float = 0.0, 
                 steering_change_weight: float = 0.0, 
                 velocity_change_weight: float= 0.0,
                 inital_velocity: float = 1.5,
                 normalize: bool = True):
        # print the args used
        print("Reward function args:")
        print("collision_penalty: ", collision_penalty)
        print("progress_weight: ", progress_weight)
        print("raceline_delta_weight: ", raceline_delta_weight)
        print("velocity_weight: ", velocity_weight)
        print("steering_change_weight: ", steering_change_weight)
        print("velocity_change_weight: ", velocity_change_weight)
        print("inital_velocity: ", inital_velocity)
        print("normalize: ", normalize)

        self.weights = [progress_weight, raceline_delta_weight, velocity_weight, steering_change_weight, velocity_change_weight]
        self.rewards = [ProgressReward, RacelineDeltaReward, VelocityReward, MinSteeringChangeReward, MinVelocityChangeReward]
        self.normalize = normalize
        self.collision_penalty = collision_penalty
        self.progress_weight = progress_weight
        self.raceline_delta_weight = raceline_delta_weight
        self.velocity_weight = velocity_weight
        self.steering_change_weight = steering_change_weight
        self.velocity_change_weight = velocity_change_weight

        self.progress_reward = ProgressReward(track)
        self.raceline_reward = RacelineDeltaReward(track)
        self.velocity_reward = VelocityReward(track)
        # print(env.action_space.low)
        self.steering_change_reward = MinSteeringChangeReward(env.action_space.low[0][0], 
                                                              env.action_space.high[0][0])
        self.velocity_change_reward = MinVelocityChangeReward(env.action_space.low[0][1], 
                                                              env.action_space.high[0][1],
                                                              inital_velocity=inital_velocity)

        # for each reward instantiate a normalization object
        self.normalizers = []
        for reward in self.rewards:
            self.normalizers.append(NormalizeReward())

    def __call__(self, obs, action, done):
        # get pose_x and pose_y from obs
        # assert action is a numpy array of length 2 if velocity change is not zero
        assert(len(action[0]) == 2 or self.velocity_change_reward == 0)
        assert ('poses_x' in obs and
        'poses_y' in obs and 
        'linear_vels_x' in obs and 
        'linear_vels_y' in obs, 
        "Some keys are missing from the obs dictionary"
        )
        
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        rewards = np.zeros(len(self.rewards))
        if self.progress_weight != 0:
            progress_reward = self.progress_reward(pose)
            rewards[0] = progress_reward
        if self.raceline_delta_weight != 0:
            raceline_reward = self.raceline_reward(pose)
            rewards[1] = raceline_reward
        if self.velocity_weight != 0:
            velocity_reward = self.velocity_reward(obs['linear_vels_x'][0], obs['linear_vels_y'][0])
            rewards[2] = velocity_reward
        if self.steering_change_reward != 0:
            steering_change_reward = self.steering_change_reward(action[0])
            rewards[3] = steering_change_reward
        if self.velocity_change_reward != 0:
            velocity_change_reward = self.velocity_change_reward(obs['linear_vels_x'][0], obs['linear_vels_y'][0]) 
            rewards[4] = velocity_change_reward
        #print(" [ProgressReward, RacelineDeltaReward, VelocityReward, MinSteeringChangeReward, MinVelocityChangeReward]")
        #print("before normalization: ", rewards)
        assert(len(self.weights) == len(self.rewards))
        if self.normalize:
            for i in range(len(self.rewards)):
                if (abs(self.weights[i]) < 0.0000001):
                    continue
                else:
                    # normalize the reward
                    rewards[i] = self.normalizers[i].step(rewards[i], done) 
        #print("after norm", rewards) # now normalized
        rewards = rewards* self.weights
        #print("after weighting", rewards)
        # sum rewards
        reward = np.sum(rewards)
        #print("rewards", reward)
        # now apply penalty for collision
        if done:
            reward = self.collision_penalty
        return reward

    def reset(self, pose: Tuple[float, float]):
        self.velocity_change_reward.reset()
        self.steering_change_reward.reset()
        # print(pose)
        self.progress_reward.reset(new_pose=pose)
        

    


        
