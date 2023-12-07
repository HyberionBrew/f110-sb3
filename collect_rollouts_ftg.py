import gymnasium
from argparse import Namespace
import yaml
import numpy as np
import torch
import pickle as pkl
from absl import flags, app

from f110_sim_env.base_env import make_base_env
import argparse
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from ftg_agents.agents_numpy import StochasticContinousFTGAgent

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
parser.add_argument('--model_path', type=str, default='/home/fabian/f110_rl/f110-sb3/logs101/progress_weight/checkpoints/f110_ppo_collision_penalty_-100.0_progress_weight_1.0_inital_velocity_1.5_save_all_rewards_True_11-09-2023-08-43-38_300000_steps.zip', help='Logging directory')
parser.add_argument('--model_name', type=str, default='progress', help='The model that was used')
parser.add_argument('--deterministic', action='store_true', default=False, help='Whether to use deterministic actions')
parser.add_argument('--speed_multiplier', type=float, default=1.0, help='Size of speed multiplier higher is slower')
parser.add_argument('--gap_blocker', type=int, default=2, help='FTG gap block')
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
                eval=False,
                use_org_reward=True,
                min_vel=0.0,
                max_vel=0.0,)
    eval_env = TimeLimit(eval_env, max_episode_steps=500)

    model = StochasticContinousFTGAgent(gap_blocker = args.gap_blocker, speed_multiplier=args.speed_multiplier) #PPO.load(args.model_path)
    model_name = str(model) #args.model_name
    episode = 0
    timesteps = 0
    with open(f"datasets0512/{model_name}", 'wb') as f:
        pass

    print("next agent")
    while timesteps < args.timesteps:

        obs, _ = eval_env.reset()
        # print(obs)
        done = False
        truncated = False
        episode += 1
        #print("Episode:", episode)
        rewards = []
        #print(timesteps)
        episode_data = []
        import time 
        start = time.time()
        actions = []
        steerings = []
        vels = []
        progress_sin = []
        progress_cos = []
        #tensor_obs = model.policy.obs_to_tensor(obs)[0]
        #print("...")
        #print(action)
        #print(tensor_obs)
        with torch.no_grad():
            # print(tensor_obs)
            #action, _, log_prob = model.policy.forward(tensor_obs, deterministic=args.deterministic)# [0]
            #action = action[0].cpu().numpy()
            action = np.array([0.0,0.0]) # just a zero zero action to get the juicy infos
        
        # eval_env.render()
        while not done and not truncated:
            # print(obs)
            timesteps += 1

            obs, reward, done, truncated, info = eval_env.step(action)
            #
            # print(obs)
            #done = True
            #truncated = True
            with torch.no_grad():
                #print("observation_model_input:", tensor_obs)
                #tensor_obs = model.policy.obs_to_tensor(obs)[0]
                 _, action, log_prob = model(obs)
                #action, _, log_prob = model.policy.forward(tensor_obs, deterministic=args.deterministic)# [0]
                # action is numpy add action dim at 0
                action = np.expand_dims(action, axis=0)
            log_prob = float(log_prob)
            # print(action)
            new_infos = dict()
            new_infos["lidar_timestamp"] = 0.0
            new_infos["pose_timestamp"] = 0.0
            if args.record:
                # record values into zarr directory
                # print(info["observations"])
                # exit()
                with open(f"datasets0512/{model_name}", 'ab') as f:
                    #if timesteps > args.timesteps:
                    #    done = True
                    #    truncated = True
                    
                    pkl.dump((action, info["observations"], 
                              float(reward), done, 
                              truncated, log_prob, 
                              timesteps, model_name, 
                              info["collision"], info["action_raw"], new_infos), f)
            #print((action, info["observations"], float(reward), done, truncated, log_prob, timesteps, model_name, info["collision"], info["action_raw"]))
            #print("---------------")
            progress_sin.append(info["observations"]["progress_sin"])
            progress_cos.append(info["observations"]["progress_cos"])
            steerings.append(info["action_raw"][0][0])
            # vels.append(info["action_raw"][0][1])
            vels.append(info["observations"]["linear_vels_x"])
            # print(action)
            rewards.append(reward)
            #print(info["observations"])
            #if timesteps == 4:
            #    exit()
            if args.render:
                eval_env.render()
                if done or truncated:
                    #print(timesteps)
                    #print("Lap done")
                    #print("R:", reward)
                    plt.plot(progress_sin)
                    plt.plot(progress_cos)
                    plt.show()
                    plt.plot(steerings)
                    plt.show()
                    plt.plot(vels)
                    plt.show()
                    plt.plot(rewards)
                    plt.show()
            # TODO! remove this
            #if timesteps > args.timesteps:
            #    break
        #TODO! remove again
        #break
        # end = time.time()
        #print("Time:", end-start)
if __name__ == "__main__":
    main(args)