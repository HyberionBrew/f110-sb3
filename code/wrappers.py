# MIT License

# Copyright (c) 2020 FT Autonomous Team One

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE RALIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from pathlib import Path

from f110_gym.envs.track import Track

from typing import Tuple
from numba import njit
import math
import collections

from gymnasium.spaces import Box
# from code.random_trackgen import create_track, convert_track

#mapno = ["Austin","BrandsHatch","Budapest","Catalunya","Hockenheim","IMS","Melbourne","MexicoCity","Montreal","Monza","MoscowRaceway",
#         "Nuerburgring","Oschersleben","Sakhir","SaoPaulo","Sepang","Shanghai","Silverstone","Sochi","Spa","Spielberg","YasMarina","Zandvoort"]



class VelocityObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space['linear_vels_x'] = Box(shape=(1,), low=-20, high=20)
        self.observation_space['linear_vels_y'] = Box(shape=(1,), low=-20, high=20)
        # self.observation_space['']
        self.observation_space['ang_vels_z'] = Box(shape=(1,), low=-20, high=20)
    def observation(self, obs):
        # clip obs between low and high
        obs['linear_vels_x'] = np.clip(obs['linear_vels_x'], -20, 20)
        obs['linear_vels_y'] = np.clip(obs['linear_vels_y'], -20, 20)
        obs['ang_vels_z'] = np.clip(obs['ang_vels_z'], -20, 20)
        # obs['linear_vels_y'] = np.clip(obs['linear_vels_y'], -20, 20)
        return obs

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip: int):
        self.frame_skip = skip
        super().__init__(env)

    def step(self, action):
        R = 0
        for t in range(self.frame_skip):
            obs, reward, done, truncate, info = self.env.step(action)
            R += reward
            if done or truncate:
                break
        return obs, R, done, truncate, info

class NormalizePose(gym.ObservationWrapper):
    def __init__(self, env, low=-30, high=30):
        super().__init__(env)
        self.observation_space['poses_x'] = Box(shape=(1,), low=low, high=high)
        self.observation_space['poses_y'] = Box(shape=(1,), low=low, high=high)
        self.observation_space['poses_theta'] = Box(shape=(1,), low=-np.pi*2, high=np.pi*2)
    def observation(self, observation):
        low = self.observation_space['poses_x'].low
        high = self.observation_space['poses_x'].high
        # clip obs between low and high
        # assert that poses are in range
        assert(observation['poses_x'] < self.observation_space['poses_x'].high and observation['poses_x'] > self.observation_space['poses_x'].low)
        assert(observation['poses_y'] < self.observation_space['poses_y'].high and observation['poses_y'] > self.observation_space['poses_y'].low)
        observation['poses_x'] = np.clip(observation['poses_x'], low, high)
        # normalise between -1 and 1
        observation['poses_x'] = 2 * ((observation['poses_x'] - low) / (high - low)) - 1

        observation['poses_y'] = np.clip(observation['poses_y'], low, high)
        # normalise between -1 and 1
        observation['poses_y'] = 2 * ((observation['poses_y'] - low) / (high - low)) - 1

        observation['poses_theta'] = np.clip(observation['poses_theta'], -np.pi*2, np.pi*2)
        # normalise between -1 and 1
        observation['poses_theta'] = 2 * ((observation['poses_theta'] + np.pi*2) / (np.pi* 4)) - 1
        return observation


class NormalizeVelocityObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        assert 'linear_vels_x' in obs
        low = self.observation_space['linear_vels_x'].low
        high = self.observation_space['linear_vels_x'].high
        # clip obs between low and high
        obs['linear_vels_x'] = np.clip(obs['linear_vels_x'], low, high)
        # normalise between -1 and 1
        obs['linear_vels_x'] = 2 * ((obs['linear_vels_x'] - low) / (high - low)) - 1

        obs['linear_vels_y'] = np.clip(obs['linear_vels_y'], low, high)
        # normalise between -1 and 1
        obs['linear_vels_y'] = 2 * ((obs['linear_vels_y'] - low) / (high - low)) - 1
        
        obs['ang_vels_z'] = np.clip(obs['ang_vels_z'], low, high)
        # normalise between -1 and 1
        obs['ang_vels_z'] = 2 * ((obs['ang_vels_z'] - low) / (high - low)) - 1
        
        assert(not(np.isnan(obs['linear_vels_y'])))
        assert(not(np.isnan(obs['linear_vels_x'])))
        return obs




class ProgressObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProgressObservation, self).__init__(env)
        self.progress_tracker = Progress(self.env.track)
        # add progress with range [0,1] to the observation space
                # extend observation space
        obs_dict = collections.OrderedDict()
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        obs_dict["progress_sin"] = Box(shape=(1,), low=-1, high=1)
        obs_dict["progress_cos"] = Box(shape=(1,), low=-1, high=1)
        self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, obs):
        assert 'poses_x' in obs
        assert 'poses_y' in obs
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        pose = pose[np.newaxis, :]
        # print(pose)
        progress = np.clip(self.progress_tracker.get_progress(pose),0,1)
        # obs['progress'] = np.array(progress, dtype=np.float32)
        # print(obs['progress'] )
        angle = 2 * np.pi *progress
        obs['progress_sin'] = np.sin(angle).astype(np.float32)
        obs['progress_cos'] = np.cos(angle).astype(np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        pose = pose[np.newaxis, :]
        
        self.progress_tracker.reset(pose)

        progress = np.clip(self.progress_tracker.get_progress(pose),0,1)
        angle = 2 * np.pi * progress
        
        obs['progress_sin'] = np.sin(angle).astype(np.float32)
        obs['progress_cos'] = np.cos(angle).astype(np.float32)
        return obs, info

class DownsampleLaserObservation(gym.ObservationWrapper):
    def __init__(self,env,max_range:float=10.0, subsample:int=20):
        super(DownsampleLaserObservation, self).__init__(env)
        
        self._max_range = max_range
        self.subsample = subsample
        obs_dict = collections.OrderedDict()
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        # print(self.observation_space["scans"].shape)
        obs_dict['lidar_occupancy'] = gym.spaces.Box(low=0, high=1, dtype=np.float32,
                                                shape=(self.observation_space["scans"].shape[1]//subsample,))
        
        self.observation_space = gym.spaces.Dict(obs_dict)
    def observation(self, observation):
        assert 'scans' in observation
        scan = observation['scans'][0]
        # clip scan
        # to max_range
        #print("scan", scan.shape)
        #print("observations, pose", observation['poses_x'], observation['poses_y'], observation['poses_theta'])
        scan = np.clip(scan,0,self._max_range)
        scan = scan / self._max_range
        observation['lidar_occupancy'] = scan[::self.subsample]
        # print(observation["lidar_occupancy"])
        return observation
    


class LidarOccupancyObservation(gym.ObservationWrapper):
    def __init__(self, env, max_range: float = 10.0, resolution: float = 0.08, degree_fow: int = 270):
        super(LidarOccupancyObservation, self).__init__(env)
        self._max_range = max_range
        self._resolution = resolution
        self._degree_fow = degree_fow
        self._n_bins = math.ceil(2 * self._max_range / self._resolution)
        # extend observation space
        obs_dict = collections.OrderedDict()
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        obs_dict['lidar_occupancy'] = gym.spaces.Box(low=0, high=255, dtype=np.uint8,
                                                     shape=(1, self._n_bins, self._n_bins))
        self.observation_space = gym.spaces.Dict(obs_dict)

    @staticmethod
    @njit(fastmath=False, cache=True)
    def _polar2cartesian(dist, angle, n_bins, res):
        occupancy_map = np.zeros(shape=(n_bins, n_bins), dtype=np.uint8)
        xx = dist * np.cos(angle)
        yy = dist * np.sin(angle)
        #print(angle.shape)
        #print(dist.shape)
        #print(xx.shape)

        xi, yi = np.floor(xx / res), np.floor(yy / res)
        for px, py in zip(xi, yi):
            row = min(max(n_bins // 2 + py, 0), n_bins - 1)
            col = min(max(n_bins // 2 + px, 0), n_bins - 1)
            if row < n_bins - 1 and col < n_bins - 1:
                # in this way, then >max_range we dont fill the occupancy map in order to let a visible gap
                occupancy_map[int(row), int(col)] = 255
        return np.expand_dims(occupancy_map, 0)

    def observation(self, observation):
        assert 'scans' in observation
        scan = observation['scans'][0]
        scan_angles = self.sim.agents[0].scan_angles  # assumption: all the lidars are equal in ang. spectrum
        # reduce fow
        mask = abs(scan_angles) <= np.deg2rad(self._degree_fow / 2.0)  # 1 for angles in fow, 0 for others
        scan = np.where(mask, scan, np.Inf)
        observation['lidar_occupancy'] = self._polar2cartesian(scan, scan_angles, self._n_bins, self._resolution)
        # print(observation["lidar_occupancy"])
        return observation


class Progress:
    def __init__(self, track: Track, lookahead: int = 20) -> None:
        # 
        xs = track.centerline.xs
        ys = track.centerline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        # append first point to end to make loop
        self.centerline = np.vstack((self.centerline, self.centerline[0]))

        self.segment_vectors = np.diff(self.centerline, axis=0)
        # print(segment_vectors.shape)
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        
        # Extend segment lengths to compute cumulative distance
        self.cumulative_lengths = np.hstack(([0], np.cumsum(self.segment_lengths)))
        self.previous_closest_idx = 0
        self.max_lookahead = lookahead
        #print(self.centerline)
        #print(self.centerline.shape)
        #print("***********")

    def distance_along_centerline_np(self, pose_points):
        assert len(pose_points.shape) == 2 and pose_points.shape[1] == 2

        # centerpoints = np.array(centerpoints)
        #print(self.centerline.shape)
        #print(centerpoints[:-1])
        #print(pose_points)
        #print(".....")
        # assert pose points must be Nx2
        pose_points = np.array(pose_points)
        #print(pose_points.shape)

        def projected_distance(pose):
            rel_pose = pose - self.centerline[:-1]
            t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
            t = np.clip(t, 0, 1)
            projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
            distances = np.linalg.norm(pose - projections, axis=1)
            points_len = self.centerline.shape[0]-1  # -1 because of last fake 
            lookahead_idx = (self.max_lookahead + self.previous_closest_idx) % points_len
            # wrap around
            if self.previous_closest_idx <= lookahead_idx:
                indices_to_check = list(range(self.previous_closest_idx, lookahead_idx + 1))
            else:
                # Otherwise, we need to check both the end and the start of the array
                indices_to_check = list(range(self.previous_closest_idx, points_len)) \
                    + list(range(0, lookahead_idx+1))
            # Extract the relevant distances using fancy indexing
            subset_distances = distances[indices_to_check]

            # Find the index of the minimum distance within this subset
            subset_idx = np.argmin(subset_distances)

            # Translate it back to the index in the original distances array
            closest_idx = indices_to_check[subset_idx]
            self.previous_closest_idx = closest_idx
            # print(closest_idx)
            return self.cumulative_lengths[closest_idx] + self.segment_lengths[closest_idx] * t[closest_idx]
        
        return np.array([projected_distance(pose) for pose in pose_points])
    
    def get_progress(self, pose: Tuple[float, float]):
        progress =  self.distance_along_centerline_np(pose)
        # print(self.cumulative_lengths.shape)
        # print(self.cumulative_lengths[-1])
        progress = progress / (self.cumulative_lengths[-1] + self.segment_lengths[-1])
        # clip between 0 and 1 (it can sometimes happen that its slightly above 1)
        # print("progress", progress)
        return np.clip(progress, 0, 1)
    
    def reset(self, pose):
        rel_pose = pose - self.centerline[:-1]
        t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
        t = np.clip(t, 0, 1)
        projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
        distances = np.linalg.norm(pose - projections, axis=1)
        
        closest_idx = np.argmin(distances)
        self.previous_closest_idx = closest_idx

def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min
"""
class FixSpeedControl(gym.ActionWrapper):
"""
    #reduce original problem to control only the steering angle
"""

    def __init__(self, env, fixed_speed: float = 2.0):
        super(FixSpeedControl, self).__init__(env)
        self._fixed_speed = fixed_speed
        print(self.action_space)
        self.action_space = #gym.spaces.Dict({'steering': self.env.action_space['steering']})

    def action(self, action):
        assert 'steering' in action
        new_action = {'steering': action['steering'], 'velocity': self._fixed_speed}
        return new_action
"""

class FixSpeedControl(gym.ActionWrapper):
    """
    reduce original problem to control only the steering angle
    """

    def __init__(self, env, fixed_speed: float = 2.0):
        super(FixSpeedControl, self).__init__(env)
        self._fixed_speed = fixed_speed
        # print(self.env.action_space)
        low = np.array([self.sim.params['s_min']]).astype(np.float32)
        high = np.array([self.sim.params['s_max']]).astype(np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

    def action(self, action):
        assert len(action.shape) == 1
        # print(action)
        new_action = np.array([[action[0], self._fixed_speed]])
        # print(new_action)
        return new_action


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        flatten = gym.spaces.utils.unflatten(self.env.action_space, action)
        return flatten

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)
    
class ProgressReward(gym.Wrapper):
    def __init__(self, env, collision_penalty = 10.0):
        super().__init__(env)
        self.progress_tracker = Progress(self.env.track)
        self.current_progress = None
        self.collision_penalty = collision_penalty

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        pose = np.array([observation['poses_x'][0], observation['poses_y'][0]])
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
        if observation['collisions'][0]:
            reward = -10

        return observation, reward, done, truncated, info


    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        pose = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        pose = pose[np.newaxis, :]
        self.progress_tracker.reset(pose)
        self.current_progress = self.progress_tracker.get_progress(pose)[0]
        return observation, info

class SpinningReset(gym.Wrapper):
    def __init__(self, env, maxAngularVel):
        super().__init__(env)
        self.maxAngularVel = maxAngularVel #TODO! replace with angular vel
    def step(self,action):
        observation, ac, done, truncated , info = self.env.step(action)
        # print(observation)
        if abs(observation['ang_vels_z'][0]) > self.maxAngularVel:
            done = True
            observation['collisions'][0] = True
        return observation, ac, done, truncated, info

from code.reward import PureProgressReward

class MaxLaps(gym.Wrapper):
    def __init__(self, env, max_laps=1, finished_reward=20, max_lap_time=1000, 
                 use_org_reward=False):
        super().__init__(env)
        if max_laps!=1:
            raise NotImplementedError # but should work, untested
        self.max_laps = max_laps
        self.finished_reward = finished_reward
        self.pure_progress_reward = PureProgressReward(env.track)
        self.timestep = 0
        self.max_lap_time = max_lap_time
        self.use_org_reward = use_org_reward

    def step(self,action):
        observation, reward, done, truncated , info = self.env.step(action)
        # pose = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        pure_progress = self.pure_progress_reward([observation['poses_x'][0], observation['poses_y'][0]])
        org_reward = reward
        reward = 0
        self.timestep += 1
        if pure_progress > self.max_laps:
            truncated = True
            # clip timestep
            self.timestep = min(self.timestep, self.max_lap_time)
            reward = self.max_lap_time - self.timestep
            if reward == 0:
                reward = 1
        if self.use_org_reward:
            reward = org_reward
        return observation, reward , done or truncated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.timestep = 0
        self.pure_progress_reward.reset(new_pose=[observation['poses_x'][0], observation['poses_y'][0]])
        return observation, info
    
class MinSpeedReset(gym.Wrapper):
    def __init__(self, env, minSpeed):
        super().__init__(env)
        self.minSpeed = minSpeed
    def step(self,action):
        observation, ac, done, truncated , info = self.env.step(action)
        # print(observation['linear_vels_x'])
        if abs(observation['linear_vels_x']) + abs(observation['linear_vels_y']) < self.maxTheta:
            done = True
        return observation, ac, done, truncated, info

class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car, but should be expanded to handle multi-agent scenarios
    """

    def __init__(self, env, fixed_speed = True, max_speed=1.0):
        super().__init__(env)
        self.progress_tracker = Progress(self.env.track)
        self.current_progress = None
        # print("init")
        #print(env.num_agents)
        # print("-------")
        self.fixed_speed = fixed_speed
        # normalised action space, steer and speed
        if self.fixed_speed:
            self.action_space = spaces.Box(low=np.array(
                [-1.0, -1.0]), high=np.array([1.0,1.0]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=np.array(
                [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # print(self.action_space)
        # normalised observations, just take the lidar scans
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1080,), dtype=np.float32)

        # store allowed steering/speed/lidar ranges for normalisation
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = self.env.params['v_min']
        self.v_max = self.env.params['v_max']
        self.lidar_min = 0
        self.lidar_max = 30  # see ScanSimulator2D max_range

        # store car dimensions and some track info
        self.car_length = self.env.params['length']
        self.car_width = self.env.params['width']
        self.track_width = 0.2  # ~= track width, see random_trackgen.py

        # radius of circle where car can start on track, relative to a centerpoint
        self.start_radius = (self.track_width / 2) - \
            ((self.car_length + self.car_width) / 2)  # just extra wiggle room

        self.step_count = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100
        self.lap = 0

    def step(self, action):
        # convert normalised actions (from RL algorithms) back to actual actions for simulator
        action_convert = self.un_normalise_actions(action)
        observation, _, done, truncated , info = self.env.step(np.array([action_convert]))

        self.step_count += 1

        # TODO -> do some reward engineering here and mess around with this
        reward = 0

        #eoins reward function
        vel_magnitude = np.linalg.norm(
            [observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        reward = vel_magnitude #/10 maybe include if speed is having too much of an effect
        
        pose = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        pose = pose[np.newaxis, :]
        new_progress = self.progress_tracker.get_progress(pose)[0]
        delta_progress = 0
        if self.current_progress > 0.9 and new_progress < 0.1:
            delta_progress = (new_progress + 1) - self.current_progress 
        else:
            delta_progress = new_progress - self.current_progress
        
        self.current_progress = new_progress
        # reward = progress
        # print(new_progress)
        if delta_progress < 0:
            # actually we could still take the delta_progres
            # why would you drive in the wrong direction hm
            reward = 0.0
        else:
            reward = delta_progress * 100



        # end episode if car is spinning
        # if abs(observation['poses_theta'][0]) > self.max_theta:
        #     done = True
        # if speed is higher than 1.5, then the car is going too fast
        vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        # reduce the reward depending on how far away from 1.0 the speed is
        # print(vel_magnitude)
        #goal = abs((vel_magnitude - 1))**2 / 50
        #print(goal)
        #print(reward)
        #reward -= goal

        if reward < 0:
            reward = 0
        if observation['collisions'][0]:
            reward = -10

        """
        if vel_magnitude > 2.0:
            reward -= vel_magnitude - 1.0
        if vel_magnitude < 1.0:
            reward = -1.0
        """
        """
        vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        #print("V:",vel_magnitude)
        if vel_magnitude > 0.2:  # > 0 is very slow and safe, sometimes just stops in its tracks at corners
            reward += 0.1"""


        # penalise changes in car angular orientation (reward smoothness)
        """ang_magnitude = abs(observation['ang_vels_z'][0])
        #print("Ang:",ang_magnitude)
        if ang_magnitude > 0.75:
            reward += -ang_magnitude/10
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 5:
            reward = -(vel_magnitude/10)

        # if collisions is true, then the car has crashed
        if observation['collisions'][0]:
            self.count = 0
            #reward = -100
            reward = -1

        # end episode if car is spinning
        if abs(observation['poses_theta'][0]) > self.max_theta:
            self.count = 0
            reward = -100
            reward = -1
            done = True

        # just a simple counter that increments when the car completes a lap
        if self.env.lap_counts[0] > 0:
            self.count = 0
            reward += 1
            if self.env.lap_counts[0] > 1:
                reward += 1
                self.env.lap_counts[0] = 0"""
        #print(done)
        #print(truncated)
        return self.normalise_observations(observation['scans'][0]), reward, bool(done), truncated, info
    
    def reset(self, seed=None, options=None): #start_xy=None, direction=None):
        # should start off in slightly different position every time
        # position car anywhere along line from wall to wall facing
        # car will never face backwards, can face forwards at an angle
        # print("reset")
        #print(options)
        #print(options["poses"])
        #print("---")
        if options is None:
            options = dict(poses=[(None),None])
        # print()
        #print(options["poses"][0])
        #print(options["poses"][1])
        start_xy = options["poses"][0]
        direction = options["poses"][1]
        # start from origin if no pose input
        if start_xy is None:
            start_xy = np.zeros(2)
        # start in random direction if no direction input
        if direction is None:
            direction = np.random.uniform(0, 2 * np.pi)
        #print(start_xy)
        #print(direction)
        # get slope perpendicular to track direction
        slope = np.tan(direction + np.pi / 2)
        # get magintude of slope to normalise parametric line
        magnitude = np.sqrt(1 + np.power(slope, 2))
        # get random point along line of width track
        rand_offset = np.random.uniform(-0.3, 0.3)
        rand_offset_scaled = rand_offset * self.start_radius

        # convert position along line to position between walls at current point
        x, y = start_xy + rand_offset_scaled * np.array([1, slope]) / magnitude

        # point car in random forward direction, not aiming at walls
        t = -np.random.uniform(max(-rand_offset * np.pi / 2, 0) - np.pi / 2,
                               min(-rand_offset * np.pi / 2, 0) + np.pi / 2) + direction
        # reset car with chosen pose
        ops = dict(poses=np.array([[x, y, t]]))
        #print(ops)
        #print(ops["poses"].shape)

        # reset progress reward
        # self.progress_tracker.reset()

        observation, info = self.env.reset(options = ops)
                
        pose = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        pose = pose[np.newaxis, :]
        self.current_progress = self.progress_tracker.get_progress(pose)[0]

        #observation = dict()
        #observation['scans'] = [[0,1]]
        # reward, done, info can't be included in the Gym format
        #print(self.observation_space)
        #print(observation["scans"][0])
        #print(observation["scans"][0].dtype)
        # to float 64
        observation["scans"][0] = np.array(observation["scans"][0], dtype=np.float64)
        return self.normalise_observations(observation['scans'][0]), info
    
    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        #if self.fixed_speed:
        #    return np.array([steer, self._fixed_speed], dtype=np.float)
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])

    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None

    def seed(self, seed):
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")

from f110_gym.envs.track import Track

class PoseStartPosition(gym.Wrapper):
    """
    Places the car on the predefined pose (handed down from reset options)
    """
    def __init__(self, env):
        super().__init__(env)
    def reset(self,seed=None, options=None):
        # print(options)
        return self.env.reset(options=options)
import warnings

class RandomStartPosition(gym.Wrapper):
    """
    Places the car in a random position on the track 
    according to the centerline
    """
    def __init__(self, env, increased = None, likelihood =None, random_pos=False):
        super().__init__(env)
        assert((increased is None and likelihood is None) or 
               (increased is not None and likelihood is not None))
        assert increased is None or len(increased) == 2, "Assertion failed: Expected length of increased to be 2"

        self.cl_x = self.env.track.centerline.xs
        self.cl_y = self.env.track.centerline.ys
        self.increased = increased
        self.likelihood = likelihood
        self.random_pos = random_pos
        # print(self.cl_x)
    # get random starting position from centerline
    
    def reset(self,seed=None, options=None):
        # print("reset!!")
        # sample according to likelihood either from the whole range or only from the increased range
        random_index = np.random.randint(len(self.cl_x))
        if self.increased is not None:
            if np.random.uniform() < self.likelihood:
                random_index = np.random.randint(self.increased[0],self.increased[1])

        # print(random_index)
        start_xy = (self.cl_x[random_index],self.cl_y[random_index])
        #print(start_xy)
        random_next = (random_index + 1) % len(self.cl_x)
        next_xy = (self.cl_x[random_next],self.cl_y[random_next])
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                                next_xy[0] - start_xy[0])
        
        # some random pertubation to the direction
        if self.random_pos:
            direction = np.random.uniform(direction - np.pi/4, direction + np.pi/4)
            # some random perputation to the position
            start_xy = (start_xy[0] + np.random.uniform(-0.1,0.1), # only small perbutation due to the specific track
                        start_xy[1] + np.random.uniform(-0.1,0.1))
            
        # print User warning if poses containted in options
        if options is None:
            options = {} 
        
        if "poses" in options: 
            warnings.warn("Only velocity will be set")   
        if "velocity" not in options:
            warnings.warn("Starting with velocity 0")
            options["velocity"] = 0.0

        reset_options = dict(poses=np.array([[start_xy[0],start_xy[1],direction]]),
                             velocity=np.array([options["velocity"]]))
        # print(reset_options)
        assert(not(np.isnan(start_xy[0])))
        assert(not(np.isnan(start_xy[0])))
        assert(not(np.isnan(direction)))
        return self.env.reset(options=reset_options)


"""
class RandomMap(gym.Wrapper):
    #Generates random maps at chosen intervals, when resetting car,
    #and positions car at random point around new track

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20

    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0

    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # create map
            for _ in range(self.MAX_CREATE_ATTEMPTS):
                try:
                    track, track_int, track_ext = create_track()
                    convert_track(track,
                                  track_int,
                                  track_ext,
                                  self.current_seed)
                    break
                except Exception:
                    print(
                        f"Random generator [{self.current_seed}] failed, trying again...")
            # update map
            self.update_map(f"./maps/map{self.current_seed}", ".png")
            # store waypoints
            self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",
                                           delimiter=',')
        # get random starting position from centerline
        random_index = np.random.randint(len(self.waypoints))
        start_xy = self.waypoints[random_index]
        print(start_xy)
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(options=dict(poses=np.array([start_xy[0],start_xy[1],direction])))

    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)

    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
"""

"""
class RandomF1TenthMap(gym.Wrapper):
 
    #Places the car in a random map from F1Tenth
    

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20

    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0

    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # update map
            randmap = mapno[np.random.randint(low=0, high=22)]
            #self.update_map(f"./maps/map{self.current_seed}", ".png")
            self.update_map(f"./f1tenth_racetracks/{randmap}/{randmap}_map", ".png")
            # store waypoints
            #self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",delimiter=',')
            self.waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
            globwaypoints = self.waypoints

        # get random starting position from centerline
        random_index = np.random.randint(len(self.waypoints))
        start_xy = self.waypoints[random_index]  #len = 4
        start_xy = start_xy[:2]
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(start_xy=start_xy, direction=direction)

    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)

    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass

    """

class ThrottleMaxSpeedReward(gym.RewardWrapper):
    """
    Slowly increase maximum reward for going fast, so that car learns
    to drive well before trying to improve speed
    """

    def __init__(self, env, start_step, end_step, start_max_reward, end_max_reward=None):
        super().__init__(env)
        # initialise step boundaries
        self.end_step = end_step
        self.start_step = start_step
        self.start_max_reward = start_max_reward
        # set finishing maximum reward to be maximum possible speed by default
        self.end_max_reward = self.v_max if end_max_reward is None else end_max_reward

        # calculate slope for reward changing over time (steps)
        self.reward_slope = (self.end_max_reward - self.start_max_reward) / (self.end_step - self.start_step)

    def reward(self, reward):
        # maximum reward is start_max_reward
        if self.step_count < self.start_step:
            return min(reward, self.start_max_reward)
        # maximum reward is end_max_reward
        elif self.step_count > self.end_step:
            return min(reward, self.end_max_reward)
        # otherwise, proportional reward between two step endpoints
        else:
            return min(reward, self.start_max_reward + (self.step_count - self.start_step) * self.reward_slope)
