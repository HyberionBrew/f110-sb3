import gymnasium
from argparse import Namespace
import yaml
import numpy as np
import torch
import pickle as pkl
from absl import flags, app

from base_env import make_base_env
import argparse
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

parser = argparse.ArgumentParser(description='Your script description')

parser.add_argument('--timesteps', type=int, default=10_000, help='Number of timesteps to run for')
parser.add_argument('--sub_sample', type=int, default=10, help='Number of lidar rays to subsample')
parser.add_argument('--agent', type=str, default='StochasticFTGAgent', help='Name of agent to use')
parser.add_argument('--map_config', type=str, default='config.yaml', help='Name of map config file')
parser.add_argument('--record', action='store_true', default=False, help='Whether to record the run')
parser.add_argument('--norender', action='store_false', default=True, dest='render', help='Whether to render the run')
parser.add_argument('--speed', type=float, default=1.0, help='Mean speed of the car')
parser.add_argument('--track', type=str, default='Infsaal', help='Track to train on')
parser.add_argument('--fixed_speed', type=float, default=None, help='Fixing the speed to the provided value')
# model path
parser.add_argument('--model_path', type=str, default='logs/models/best_model.zip', help='Logging directory')
parser.add_argument('--model_name', type=str, default='progress', help='The model that was used')
args = parser.parse_args()

eval_config = {
    "collision_penalty": -10.0,
    "progress_weight": 1.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "inital_velocity": 1.5,
    "normalize": False,
}

import matplotlib.pyplot as plt
import zarr
import pickle as pkl

def main(args):
    eval_env = make_base_env(map= args.track,
                fixed_speed=args.fixed_speed,
                random_start =True,
                train_random_start = False,
                reward_config = eval_config,
                eval=True,
                use_org_reward=True,)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)

    model = PPO.load(args.model_path)
    model_name = args.model_name
    episode = 0
    timesteps = 0
    with open(f"datasets5/{args.model_name}", 'wb') as f:
        pass


    while timesteps < args.timesteps:

        obs, _ = eval_env.reset()
        done = False
        truncated = False
        episode += 1
        print("Episode:", episode)
        rewards = []
        print(timesteps)
        episode_data = []
        import time 
        start = time.time()
        actions = []
        steerings = []
        vels = []
        tensor_obs = model.policy.obs_to_tensor(obs)[0]
        #print("...")
        #print(action)
        #print(tensor_obs)
        with torch.no_grad():
            print(tensor_obs)
            action, _, log_prob = model.policy.forward(tensor_obs, deterministic=True)# [0]
        obs, reward, done, truncated, info = eval_env.step(action)
        while not done and not truncated:
            # print(obs)
            timesteps += 1
            # remove key poses_theta from obs
            # del obs["poses_theta"] # TODO! remove bandaid
            # action, _ = model.predict(obs) #deterministic=True)
            tensor_obs = model.policy.obs_to_tensor(obs)[0]
            #print("...")
            #print(action)
            #print(tensor_obs)
            with torch.no_grad():
                print(tensor_obs)
                action, _, log_prob = model.policy.forward(tensor_obs, deterministic=True)# [0]
            print(action)
            action = action.squeeze(0).detach().numpy()
            log_prob = float(log_prob.detach().numpy()[0])
            # print(action)

            if args.record:
                # record values into zarr directory
                # print(info["observations"])
                # exit()
                with open(f"datasets5/{args.model_name}", 'ab') as f:
                    #if timesteps > args.timesteps:
                    #    done = True
                    #    truncated = True
                    pkl.dump((action, info["observations"], float(reward), done, truncated, log_prob, timesteps, model_name, info["collision"]), f)



            obs, reward, done, truncated, info = eval_env.step(action)
            # print(obs)
            # print(info)
            steerings.append(info["action_raw"][0][0])
            vels.append(info["action_raw"][0][1])
            # print(action)
            rewards.append(reward)

            if args.render:
                eval_env.render()
                if done or truncated:
                    #print(timesteps)
                    #print("Lap done")
                    #print("R:", reward)
                    plt.plot(steerings)
                    plt.show()
                    plt.plot(vels)
                    plt.show()
                    plt.plot(rewards)
                    plt.show()
            # TODO! remove this
            #if timesteps > args.timesteps:
            #    break
        # end = time.time()
        #print("Time:", end-start)
if __name__ == "__main__":
    main(args)