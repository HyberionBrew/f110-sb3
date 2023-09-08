from base_env import make_base_env
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import argparse

parser = argparse.ArgumentParser(description='Your script description')

parser.add_argument('--model_path', type=str, default='logs/models/best_model.zip', help='Logging directory')
parser.add_argument('--track', type=str, default='Infsaal', help='Track to train on')
parser.add_argument('--fixed_speed', type=float, default=None, help='Fixing the speed to the provided value')

args = parser.parse_args()
import matplotlib.pyplot as plt
import f110_gym
import f110_orl_dataset
import gymnasium as gym
from f110_orl_dataset import normalize_dataset

standard_config = {
    "collision_penalty": -50.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 1.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "inital_velocity": 1.5,
    "normalize": False,
}

obs_dictionary_keys = [
    "poses_x",
    "poses_y",
    "poses_theta",
    "ang_vels_z",
    "linear_vels_x",
    "linear_vels_y",
    "previous_action",
    "progress"
]
import numpy as np

def evaluate(args):
    dataset = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1,
            params=dict(vmin=0.5, vmax=2.0)),
              render_mode="human")
    )
    eval_env = make_base_env(map= args.track,
                    fixed_speed=args.fixed_speed,
                    random_start =True,
                    reward_config = standard_config,
                    eval=False)
    # eval_env = TimeLimit(eval_env, max_episode_steps=500)
    
    model = PPO.load(args.model_path, env=eval_env)
    episode = 0
    normalizer = normalize_dataset.Normalize()
    while episode < 2000:
        episode += 1
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        rew = 0
        steps = 0
        rewards = []
        while not done and not truncated:
            # print(obs)
            steps += 1
            action, _ = model.predict(obs)
            # print(action)
            # print(action.shape)
            obs, reward, done, truncated, info = eval_env.step(action)
            # print(reward)
            # print(obs)
            # print(action)
            #print(info)
            # print(info)
            # print(action)
            data_dict = info
            arrays_to_concat = [data_dict['observations'][key].reshape([data_dict['observations'][key].shape[0], -1]) for key in obs_dictionary_keys]
    
            # print the shapes of all arrays to concatenate
            # print([arr.shape for arr in arrays_to_concat])
            # Concatenate all arrays along the last dimension
            # remains to test laser
            concatenated_obs = np.concatenate(arrays_to_concat, axis=-1)
            
            concatenated_obs = np.concatenate([concatenated_obs, concatenated_obs])
            print(concatenated_obs)
            print(concatenated_obs.shape)
            print("#####")
            print(done)
            print(obs)
            unfl = normalizer.unflatten_batch(concatenated_obs)
            re_normalized = normalizer.normalize_obs_batch(unfl)
            # print(dataset.sim)
            laser_scans = dataset.get_laser_scan(concatenated_obs, 20)
            print(laser_scans.shape)
            normalized_laser_scans = normalizer.normalize_laser_scan(laser_scans)
            print(normalized_laser_scans[0])
            print(obs.keys())
            print(obs["lidar_occupancy"])
            # There is a bug, probably in F110 env, that causes the theta to be wrong on crash/done
            assert  done or np.isclose(obs["lidar_occupancy"], normalized_laser_scans[0], rtol=0.15).all()
            #print(unfl)
            #print("++++++++++++")
            
            #print("obs", obs)
            # print("info", concatenated_obs)
            # print(eval_env.observation_space)
            rewards.append(reward)
            # print("obs reward", reward)
            rew += reward * 0.99 ** steps
            if truncated:
                print(steps)
                print("info")
                print("Truncated")

            if done or truncated or steps==500:
                print(steps)
                print("Lap done")
                print("R:", rew)
                plt.plot(rewards)
                plt.show()
            eval_env.render()

if __name__ == "__main__":
    evaluate(args)