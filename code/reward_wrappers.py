import gymnasium as gym

from f110_gym.envs.track import Track

from typing import Tuple
import numpy as np

class Progress:
    def __init__(self, track: Track) -> None:
        # 
        xs = track.centerline.xs
        ys = track.centerline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        print(self.centerline)
        print(self.centerline.shape)
        print("***********")

    def distance_along_centerline_np(self, centerpoints, pose_points):
        centerpoints = np.array(centerpoints)
        pose_points = np.array(pose_points)
        
        # Calculate segment vectors and their magnitudes
        segment_vectors = np.diff(centerpoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Extend segment lengths to compute cumulative distance
        cumulative_lengths = np.hstack(([0], np.cumsum(segment_lengths)))
        
        def projected_distance(pose):
            rel_pose = pose - centerpoints[:-1]
            t = np.sum(rel_pose * segment_vectors, axis=1) / np.sum(segment_vectors**2, axis=1)
            t = np.clip(t, 0, 1)
            projections = centerpoints[:-1] + t[:, np.newaxis] * segment_vectors
            distances = np.linalg.norm(pose - projections, axis=1)
            
            closest_idx = np.argmin(distances)
            return cumulative_lengths[closest_idx] + segment_lengths[closest_idx] * t[closest_idx]
        
        return np.array([projected_distance(pose) for pose in pose_points])
    
    def get_progress(self, pose: Tuple[float, float]):
        return self.distance_along_centerline_np(self.centerline, pose)
    
    #def get_id_closest_point2centerline(self, point: Tuple[float, float], min_id: int=0):
    #    idx = (np.linalg.norm(self.centerline[min_id:, 0:2] - point, axis=1)).argmin()
    #    return idx
    #def get_progress(self, point: Tuple[float, float], above_val: float = 0.0):
    #    """ get progress by looking the closest waypoint with at least `above_val` progress """
    #    assert 0 <= above_val <= 1, f'progress must be in 0,1 (instead given above_val={above_val})'
    #    n_points = self.centerline.shape[0]
    #    min_id = int(above_val * n_points)
    #    idx = self.get_id_closest_point2centerline(point, min_id=min_id)
    #    progress = idx / n_points
    #    assert 0 <= progress <= 1, f'progress out of bound {progress}'
    #    return progress

class ProgressReward(gym.RewardWrapper):
    def __init__(self, env, collision_penalty: float = 0.0):
        assert collision_penalty >= 0.0, f"penalty must be >=0 and will be subtracted to the reward ({collision_penalty}"
        self._collision_penalty = collision_penalty
        super(ProgressReward, self).__init__(env)
        self._current_progress = None
        track = env.track
        self.P = Progress(track)

    def reset(self, seed=None, options=None):
        obs, info = super(ProgressReward, self).reset(seed=None, options=None)
        # print(obs)
        point = obs['pose'][0:2]
        self._current_progress = self.P.get_progress(point)
        return obs, info

    def _compute_progress(self, obs, info):
        assert 'pose' in obs and 'lap_count' in info
        point = obs['pose'][0:2]
        progress = info['lap_count'] + self.P.get_progress(point)
        return progress

    def step(self, action):
        obs, _, done,truncated, info = super(ProgressReward, self).step(action)
        new_progress = self._compute_progress(obs, info)
        delta_progress = new_progress - self._current_progress
        if done:    # collision
            reward = - self._collision_penalty
        elif delta_progress < 0:  # mitigate issue with progress-computation when crossing finish line
            reward = 0.0
        else:
            reward = 100 * delta_progress
        self._current_progress = new_progress
        info['progress'] = self._current_progress  # add progress info
        return obs, reward, done, truncated, info
