import numpy as np

import gymnasium as gym

# from racing_rl.envs.track import Track
from f110_gym.envs import F110Env
from f110_gym.envs.rendering import EnvRenderer
from typing import Dict, Any
import collections

class ActionDictWrapper(gym.ActionWrapper):
    def __init__(self,env, fixed_speed=None):
        super(ActionDictWrapper, self).__init__(env)
        
        steering_low, steering_high = self.sim.params['s_min'], self.sim.params['s_max']
        velocity_low, velocity_high = 0.5, 10.0
        self.fixed_speed = fixed_speed
        print(fixed_speed)
        if fixed_speed is None:
            self.action_space =  gym.spaces.Dict({
                "steering": gym.spaces.Box(low=steering_low, high=steering_high, shape=()),
                "velocity": gym.spaces.Box(low=velocity_low, high=velocity_high, shape=())
            })
        else:
            self.action_space = gym.spaces.Dict({
                "steering": gym.spaces.Box(low=steering_low, high=steering_high, shape=()),
            })
        
    def action(self, action):
        assert 'steering' in action
        if self.fixed_speed is not None:
            new_action = {'steering': action['steering'], 'velocity': self.fixed_speed}
        else:
            new_action = action
        return new_action

class RaceWrapper(gym.Wrapper):
    """
    This class implement a base wrapper to the original f1tenth_gym environment tailored on Single-Agent race,
    introducing the following changes:
        - observation and action spaces defined as dictionary with low/high limits of each observable quantities
        - automatic reset of the agent initial position
        - fix rendering issue based on map filepath
    """

    def __init__(self,env):
        super().__init__(env)
        # self.add_render_callback(render_callback)
        self._scan_size = self.sim.agents[0].scan_simulator.num_beams
        self._scan_range = self.sim.agents[0].scan_simulator.max_range
        self._step = 0
        
        self.cl_x = self.track.centerline.xs
        self.cl_y = self.track.centerline.ys
        self.centerline = np.array([self.cl_x, self.cl_y]).T
        print(self.centerline.shape)
        self.observation_space = gym.spaces.Dict({
            'scan': gym.spaces.Box(low=0.0, high=self._scan_range, shape=(self._scan_size,)),
            'pose': gym.spaces.Box(low=np.NINF, high=np.PINF, shape=(3,)),
            'velocity': gym.spaces.Box(low=-5, high=20, shape=(1,)),
        })
        steering_low, steering_high = self.sim.params['s_min'], self.sim.params['s_max']
        velocity_low, velocity_high = 0.5, 10.0     # be careful with 0 velocity, it could cause division-by-zero
        self.action_space =  gym.spaces.Dict({
            "steering": gym.spaces.Box(low=steering_low, high=steering_high, shape=()),
            "velocity": gym.spaces.Box(low=velocity_low, high=velocity_high, shape=())
        })
    @property
    def action_space(self):
        """
            steering: desired steering angle (rad)
            velocity: desired velocity (m/s)
        """
        return self._action_space
    @property
    def _default_sim_params(self):
        return {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
                'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
                'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

    @staticmethod
    def _get_flat_action(action: Dict[str, float]):
        assert 'steering' in action and 'velocity' in action
        flat_action = np.array([[action['steering'], action['velocity']]])
        assert flat_action.shape == (1, 2), f'the actions dict-array conversion returns wrong shape {flat_action.shape}'
        return flat_action

    def _prepare_obs(self, old_obs):
        assert all([f in old_obs for f in ['scans', 'poses_x', 'poses_y', 'poses_theta',
                                           'linear_vels_x', 'linear_vels_y', 'ang_vels_z',
                                           'collisions']]), f'obs keys are {old_obs.keys()}'
        # Note: the original env returns `scan` values > `max_range`. To keep compatibility wt obs-space, we clip it
        obs = collections.OrderedDict(
            {'scan': np.clip(old_obs['scans'][0], 0, self._scan_range),
             'pose': np.array([old_obs['poses_x'][0], old_obs['poses_y'][0], old_obs['poses_theta'][0]]),
             'velocity': np.array([old_obs['linear_vels_x'][0]])})
        return obs

    def _prepare_info(self, old_obs, action, old_info):
        assert all([f in old_obs for f in ['lap_times', 'lap_counts', 'collisions', 'linear_vels_x']]), f'obs keys are {old_obs.keys()}'
        assert all([f in action for f in ['steering', 'velocity']]), f'action keys are {action.keys()}'
        assert all([f in old_info for f in ['checkpoint_done']]), f'info keys are {old_info.keys()}'
        info = {'checkpoint_done': old_info['checkpoint_done'][0],
                'lap_time': old_obs['lap_times'][0],
                'lap_count': old_obs['lap_counts'][0],
                'collision': old_obs['collisions'][0],
                'velocity': old_obs['linear_vels_x'][0],
                'action': action
                }
        return info

    def step(self, action):
        """
        Note: `step` is used in the `reset` method of the original environment, to initialize the simulators
        For this reason, we cannot rid off the original fields in the observation completely.
        We overcome this by checking if the action is an array (then called in the reset) or a dictionary (otherwise)
        """
        if type(action) == np.ndarray:
            obs, reward, done, truncated, info = super().step(action)
        else:
            flat_action = self._get_flat_action(action)
            original_obs, reward, done,truncated, original_info = super().step(flat_action)
            obs = self._prepare_obs(original_obs)
            info = self._prepare_info(original_obs, action, original_info)
            done = bool(done)
        if self._gui and self._step % self._render_freq == 0:
            self.render()
        self._step += 1
        return obs, reward, done,truncated, info

    def reset(self, mode: str = 'grid'):
        """ reset modes:
                - grid: reset the agent position on the first waypoint
                - random: reset the agent position on a random waypoint
        """
        assert mode in ['grid', 'random']
        if mode == "grid":
            waypoint_id = 0
        elif mode == "random":
            waypoint_id = np.random.randint(self.centerline.shape[0] - 1)
        else:
            raise NotImplementedError(f"reset mode {mode} not implemented")
        # compute x, y, theta
        assert 0 <= waypoint_id < self.centerline.shape[0] - 1
        wp, next_wp = self.centerline[waypoint_id], self.centerline[waypoint_id + 1]
        theta = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])
        pose = [wp[0], wp[1], theta]
        # call original method
        original_obs, original_info = super().reset(options=dict(poses=np.array([pose])))
        obs = self._prepare_obs(original_obs)
        self._step = 0
        return obs

        