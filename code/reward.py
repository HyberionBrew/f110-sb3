import numpy as np
from f110_gym.envs.track import Track
from code.wrappers import Progress
from typing import Tuple
import gymnasium as gym

from gymnasium.wrappers.normalize import RunningMeanStd, update_mean_var_count_from_moments

# adapted slightly from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
class NormalizeReward():
        def __init__(self, env,
                gamma: float = 0.99,
                epsilon: float = 1e-8,):
            # TODO! Make this right
            # self.num_envs = getattr(env, "num_envs", 1)
            # self.is_vector_env = getattr(env, "is_vector_env", False)
            # print("self.num_envs", self.num_envs)
            # print("self.is_vector_env", self.is_vector_env)
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

class PureProgressReward(object):
    def __init__(self, track:Track) -> None:
        self.inital_progress = None
        self.previous_progress = None
        self.progress_tracker = Progress(track)
        self.laps = 0
        self.i = 0
    def reset(self, new_pose: Tuple[float, float]):
        pose = np.array(new_pose)
        # print(pose)
        pose = pose[np.newaxis, :]
        self.progress_tracker.reset(pose)
        self.inital_progress = self.progress_tracker.get_progress(pose)[0] #always 0
        self.previous_progress = self.inital_progress
        self.laps = 0
        # print("Reset called")

    def __call__(self, pose: Tuple[float, float]):
        pose = np.array(pose)
        pose = pose[np.newaxis, :]
        new_progress = self.progress_tracker.get_progress(pose)[0]
        #print(".....")
        #print(new_progress)
        #print(self.laps)
        #print(self.previous_progress)
        #print(self.inital_progress)
        #print(self.previous_progress)
        if (new_progress - self.previous_progress) < -0.9 \
            and new_progress < 0.1 and self.previous_progress > 0.9:
            # crossed finish line
            self.laps += 1
            # exit(0)
        reward = new_progress + self.laps - self.inital_progress
        self.previous_progress = new_progress
        return reward


class ProgressReward(object):
    def __init__(self, track:Track) -> None:
        self._current_progress = None
        self.progress_tracker = Progress(track)
    
    def reset(self, new_pose: Tuple[float, float]):
        pose = np.array(new_pose)
        # print(pose)
        pose = pose[np.newaxis, :]
        self.progress_tracker.reset(pose)
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
@brief Minimum Lidar Ray Reward
"""
class MinLidarRayReward(object):
    def __init__(self, high=1.5):
        self.high = high
    def __call__(self, laser_scan) -> float:
        # find minimum lidar ray
        min_ray = np.min(laser_scan)
        # normalize the min_distance
        # clip it to the high value
        min_ray = np.clip(min_ray, 0, self.high)
        reward = min_ray / self.high
        min_ray = min_ray ** 2 
        return reward

"""
@brief     Reward function based on the distance from the raceline
"""
class RacelineDeltaReward(object):
    def __init__(self, track: Track) -> None:
        xs = track.raceline.xs
        ys = track.raceline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        self.largest_delta_observed = 2.0

    def __call__(self, pose: Tuple[float, float]) -> float:
        # calculate the squared distances to all points on the raceline
        distances = np.sum((self.centerline - np.array(pose))**2, axis=1)
        # get the minimum squared distance
        min_distance_squared = np.min(distances)
        # take the square root to get the actual minimum distance
        min_distance = np.sqrt(min_distance_squared)
        if min_distance > self.largest_delta_observed:
            self.largest_delta_observed = min_distance

        # normalize the min_distance
        reward = min_distance / self.largest_delta_observed
        # the negative of the distance, meaning closer to the line is better
        reward = 1.0 - reward
        reward = reward ** 2
        return reward
"""
@brief    Reward function based on the change in action
"""
class MinActionReward(object):    
    def __init__(self, low_steering, high_steering):
        self.low_steering = low_steering
        self.high_steering = high_steering
    
    def __call__(self, action: np.ndarray) -> float:
        # penalize the change in steering angle
        assert(len(action) == 2)
        delta_steering = abs((action[0]))
        normalized_steering = (delta_steering / self.high_steering)**2 # prefer smaller shifts
        inverse_steering = (1.0 - normalized_steering)
        reward = inverse_steering
        return reward
    

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
        delta = abs((action[0] - self._last_action))
        #print ("delta", delta)

        normalized = delta / (self.high - self.low)
        #print(self.high, self.low)
        inverse = 1.0 - normalized
        delta = (inverse) **2 # prefer smaller shifts
        # delta = np.clip(delta, self.low, self.high)
        reward = delta
        self._last_action = action[0]
        #print(reward)
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
        delta = abs((velocity - self._last_velocity))
        normalized = delta / (self.high - self.low)
        inverse = 1.0 - normalized
        reward = (inverse) **2 # prefer smaller shifts
        self._last_velocity = velocity
        # print("vel")
        # print(delta)
        # print(reward)
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
                 min_action_weight: float = 0.0,
                 min_lidar_ray_weight: float = 0.0,
                 pure_progress_weight: float = 0.0,
                 inital_velocity: float = 1.5,
                 normalize: bool = True ,
                 ):
        # print the args used
        print("Reward function args:")
        print("collision_penalty: ", collision_penalty)
        print("progress_weight: ", progress_weight)
        print("raceline_delta_weight: ", raceline_delta_weight)
        print("velocity_weight: ", velocity_weight)
        print("steering_change_weight: ", steering_change_weight)
        print("velocity_change_weight: ", velocity_change_weight)
        print("min_action_weight", min_action_weight)
        print("min_lidar_ray_weight", min_lidar_ray_weight)
        print("inital_velocity: ", inital_velocity)
        print("normalize: ", normalize)
        self.weights = [progress_weight, raceline_delta_weight, velocity_weight, steering_change_weight, velocity_change_weight, 
                        pure_progress_weight, min_action_weight, min_lidar_ray_weight]
        self.rewards = [ProgressReward, RacelineDeltaReward, VelocityReward, 
                        MinSteeringChangeReward, MinVelocityChangeReward, 
                        PureProgressReward, MinActionReward, MinLidarRayReward]
        self.rewards_name = ["ProgressReward", "RacelineDeltaReward", 
                             "VelocityReward", "MinSteeringChangeReward", 
                             "MinVelocityChangeReward", "PureProgressReward", 
                             "MinActionReward", "MinLidarRayReward"]
        self.normalize = normalize
        self.collision_penalty = collision_penalty
        self.progress_weight = progress_weight
        self.raceline_delta_weight = raceline_delta_weight
        self.velocity_weight = velocity_weight
        self.steering_change_weight = steering_change_weight
        self.velocity_change_weight = velocity_change_weight
        self.pure_progress_weight = pure_progress_weight
        self.min_action_weight = min_action_weight
        self.min_lidar_ray_weight = min_lidar_ray_weight

        self.progress_reward = ProgressReward(track)
        self.raceline_reward = RacelineDeltaReward(track)
        self.velocity_reward = VelocityReward(track)
        self.pure_progress_reward = PureProgressReward(track)
        # print(env.action_space.low)
        self.steering_change_reward = MinSteeringChangeReward(env.action_space.low[0][0], 
                                                              env.action_space.high[0][0])
        self.velocity_change_reward = MinVelocityChangeReward(env.action_space.low[0][1], 
                                                              env.action_space.high[0][1],
                                                              inital_velocity=inital_velocity)
        self.min_action_reward = MinActionReward(env.action_space.low[0][0],
                                                    env.action_space.high[0][0])
        
        self.min_lidar_ray_reward = MinLidarRayReward()

        # for each reward instantiate a normalization object
        self.normalizers = []
        for reward in self.rewards:
            self.normalizers.append(NormalizeReward(env))

    def __call__(self, obs, action, collision, done):
        # get pose_x and pose_y from obs
        # assert action is a numpy array of length 2 if velocity change is not zero
        assert (len(action[0]) == 2 or self.velocity_change_reward == 0)
        assert ('poses_x' in obs and
                'poses_y' in obs and 
                'linear_vels_x' in obs and 
                'linear_vels_y' in obs and
                'scans' in obs), "Some keys are missing from the obs dictionary"

        
        
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        rewards = np.zeros(len(self.rewards))
        
        #print(rewards)
        #print(self.weights)
        if self.progress_weight > 0.0001:
            progress_reward = self.progress_reward(pose)
            # print("progress_reward", progress_reward)
            rewards[0] = progress_reward
        if self.raceline_delta_weight > 0.0001:
            raceline_reward = self.raceline_reward(pose)
            rewards[1] = raceline_reward
        if self.velocity_weight > 0.0001:
            velocity_reward = self.velocity_reward(obs['linear_vels_x'][0], obs['linear_vels_y'][0])
            rewards[2] = velocity_reward
        if self.steering_change_weight > 0.0001:
            steering_change_reward = self.steering_change_reward(action[0])
            rewards[3] = steering_change_reward
        if self.velocity_change_weight > 0.0001:
            velocity_change_reward = self.velocity_change_reward(obs['linear_vels_x'][0], obs['linear_vels_y'][0]) 
            rewards[4] = velocity_change_reward
        if self.pure_progress_weight > 0.0001:
            pure_progress_reward = self.pure_progress_reward(pose)
            rewards[5] = pure_progress_reward
        if self.min_action_weight > 0.0001:
            min_action_reward = self.min_action_reward(action[0])
            rewards[6] = min_action_reward
        if self.min_lidar_ray_weight > 0.0001:
            min_lidar_ray_reward = self.min_lidar_ray_reward(obs['scans'][0])
            rewards[7] = min_lidar_ray_reward
        #print(" [ProgressReward, RacelineDeltaReward, VelocityReward, MinSteeringChangeReward, MinVelocityChangeReward]")
        #print("before normalization: ", rewards)
        assert(len(self.weights) == len(self.rewards))
        if self.normalize:
            for i in range(len(self.rewards)):
                if (abs(self.weights[i]) < 0.0001):
                    continue
                else:
                    # normalize the reward
                    rewards[i] = self.normalizers[i].step(rewards[i], done) 
        #print("after norm", rewards) # now normalized
        rewards = rewards* self.weights
        #print("after weighting", rewards)
        # sum rewards
        reward = np.sum(rewards)
        # create dict from rewards
        rewards_dict = {}
        for i, rew_name in enumerate(self.rewards_name):
            rewards_dict[str(rew_name)] = rewards[i]
        #print("rewards", reward)
        # now apply penalty for collision
        if collision:
            # print("Collision Penalty")
            reward = self.collision_penalty
        # print(rewards_dict)
        return reward , rewards_dict

    def reset(self, pose: Tuple[float, float]):
        self.velocity_change_reward.reset()
        self.steering_change_reward.reset()
        # print(pose)
        self.progress_reward.reset(new_pose=pose)
        self.pure_progress_reward.reset(new_pose=pose)

    


        
