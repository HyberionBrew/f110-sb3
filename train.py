from absl import flags, app
from functools import partial
from base_env import make_base_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

from stable_baselines3 import PPO
import numpy as np
import argparse
import json
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
#TODO! normalization is only applied process wise, not across processes (i think)

parser = argparse.ArgumentParser(description='Train a model on the F1Tenth Gym environment')
parser.add_argument('--logdir', type=str, default='logs', help='Logging directory')
parser.add_argument('--track', type=str, default='Infsaal', help='Track to train on')
parser.add_argument('--fixed_speed', type=float, default=None, help='Fixing the speed to the provided value')
parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes')
parser.add_argument('--reward' , type=str, default="TD", help='Reward function to use')
parser.add_argument('--model_path', type=str, default=None, help='Path to model to load')
parser.add_argument('--progress_weight', type=float, default=0.0, help='Weight of progress reward')
parser.add_argument('--raceline_delta_weight', type=float, default=0.0, help='Weight of raceline delta reward')
parser.add_argument('--velocity_weight', type=float, default=0.0, help='Weight of velocity reward')
parser.add_argument('--steering_change_weight', type=float, default=0.0, help='Weight of steering change reward')
parser.add_argument('--velocity_change_weight', type=float, default=0.0, help='Weight of velocity change reward')
parser.add_argument('--pure_progress_weight', type=float, default=0.0, help='Weight of pure progress reward')
parser.add_argument('--min_action_weight', type=float, default=0.0, help='Weight of min action reward')
parser.add_argument('--min_lidar_ray_weight', type=float, default=0.0, help='Weight of min lidar ray reward')
parser.add_argument('--inital_velocity', type=float, default=1.5, help='Inital velocity of the car')
parser.add_argument('--normalize', type=bool, default=False, help='Normalize the reward')


args = parser.parse_args()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def save_parameters_to_file(logdir, args, filename="params.json"):
    """
    Saves the provided argparse arguments to a JSON file.

    :param args: Parsed argparse arguments
    :param filename: Name of the output file (default is "params.json")
    """
    params = vars(args)  # Convert argparse.Namespace to dict
    with open(f"{logdir}/{filename}", 'w') as file:
        json.dump(params, file, indent=4)

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
         super(RewardLoggerCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        # Access the environment using self.locals['env']
        # Example: logging the current episode reward
        # (this assumes the reward is stored in your custom Gym environment as 'current_reward')
        info = self.locals.get("infos")
        custom_reward = info[0].get('rewards', None)
        # print(custom_reward)
        for key in custom_reward.keys():
            self.logger.record_mean(f"train/{key}", custom_reward[key])

        
        # Return True to continue training, False to stop
        return True
    
import datetime
def train(args):
    
    reward_config = {
        "collision_penalty": -50.0,
        "progress_weight": args.progress_weight, # 1.0
        "raceline_delta_weight": args.raceline_delta_weight, # 0.5
        "velocity_weight": args.velocity_weight, # 0.5
        "steering_change_weight": args.steering_change_weight, # 0.5
        "velocity_change_weight": args.velocity_change_weight, # 0.5
        "pure_progress_weight": args.pure_progress_weight, # 0.5
        "min_action_weight" : args.min_action_weight,
        "min_lidar_ray_weight" : args.min_lidar_ray_weight,
        "inital_velocity": args.inital_velocity, # 1.5
        "normalize": False,
        "save_all_rewards": True,
    }

    reward_config_eval = reward_config
    """{
        "collision_penalty": -10.0,
        "progress_weight": 1.0,
        "raceline_delta_weight": 0.0,
        "velocity_weight": 0.0,
        "steering_change_weight": 0.0,
        "velocity_change_weight": 0.0,
        "pure_progress_weight": 0.0,
        "inital_velocity": 1.5,
        "normalize": False,
    }"""

    # build filename string based on reward config
    # if 0 dont include in filename else include with value
    filename = ""
    for key in reward_config.keys():
        if reward_config[key] != 0:
            filename += f"{key}_{reward_config[key]}_"
    # append the current time and date
    filename += datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    save_parameters_to_file(args.logdir, args, filename=filename + ".json")
    # create logdir
    #train_env = make_base_env(map= args.track,
    #                fixed_speed=args.fixed_speed,
    #                random_start =True,
    #                reward_config = reward_config)
    checkpoint_callback = CheckpointCallback(save_freq=100_000 // args.num_processes, 
                                             save_path=f"{args.logdir}/checkpoints/",
                                             name_prefix=f"f110_ppo_{filename}")

    partial_make_base_env = partial(make_base_env, 
                                    map=args.track,
                                    fixed_speed=args.fixed_speed,
                                    random_start =True,
                                    reward_config = reward_config)
    train_envs = make_vec_env(partial_make_base_env,
                n_envs=args.num_processes,
                seed=np.random.randint(pow(2, 31) - 1),
                vec_env_cls=SubprocVecEnv)
    
    
    

    eval_env = make_base_env(map= args.track,
                    fixed_speed=args.fixed_speed,
                    random_start =True,
                    reward_config = reward_config_eval,
                    eval=True)
           

    eval_env = Monitor(eval_env)
    # eval_env = TimeLimit(eval_env, max_episode_steps=500)
    # eval_env = Monitor(eval_env, args.logdir)
    # eval_env = RecordVideo(eval_env, f"{args.logdir}/videos", episode_trigger = lambda episode_number: True)
    eval_freq = 25_000
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(f"{args.logdir}/models"), n_eval_episodes=10,
                                 log_path=str(f"{args.logdir}/evals"), eval_freq=eval_freq//args.num_processes,
                                 deterministic=True, render=False)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_path is not None: 
        model = PPO.load(args.model_path, env=train_envs, device=device, tensorboard_log=args.logdir)
    else:
        model = PPO("MultiInputPolicy", train_envs, verbose=1, device=device,learning_rate=0.0001, tensorboard_log=args.logdir)

    # model = PPO("MultiInputPolicy", train_envs, verbose=1, device=device, tensorboard_log=args.logdir)
    # if load model
    
    model.learn(total_timesteps=200_000, callback=[eval_callback,RewardLoggerCallback(),checkpoint_callback], progress_bar=True) #, callback=eval_callback)
    # save the model
    model.save(f"{args.logdir}/models/f110_ppo_final_{filename}")
    # model.learn(total_timesteps=500_000, callback=[eval_callback], progress_bar=True) #, callback=eval_callback)
if __name__ == "__main__":
    train(args)