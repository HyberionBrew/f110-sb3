from absl import flags, app
from functools import partial
from base_env import make_base_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

from stable_baselines3 import PPO
import numpy as np
import argparse
import json
import torch
from stable_baselines3.common.monitor import Monitor


parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--logdir', type=str, default='logs', help='Logging directory')
parser.add_argument('--track', type=str, default='Infsaal', help='Track to train on')
parser.add_argument('--fixed_speed', type=float, default=None, help='Fixing the speed to the provided value')
parser.add_argument('--num_process', type=int, default=1, help='Number of parallel processes')
parser.add_argument('--reward' , type=str, default="TD", help='Reward function to use')
args = parser.parse_args()

def save_parameters_to_file(logdir, args, filename="params.json"):
    """
    Saves the provided argparse arguments to a JSON file.

    :param args: Parsed argparse arguments
    :param filename: Name of the output file (default is "params.json")
    """
    params = vars(args)  # Convert argparse.Namespace to dict
    with open(filename, 'w') as file:
        json.dump(params, file, indent=4)

def train(args):

    save_parameters_to_file(args.logdir, args)
    # create logdir
    train_env = make_base_env(map= args.track,
                    fixed_speed=args.fixed_speed,
                    random_start =True,
                    reward = args.reward)
    
    train_env = TimeLimit(train_env, max_episode_steps=1000)
    
    eval_env = make_base_env(map= args.track,
                    fixed_speed=args.fixed_speed,
                    random_start =True,
                    reward="TD")
    eval_env = Monitor(eval_env)
    eval_env = TimeLimit(eval_env, max_episode_steps=500)
    # eval_env = Monitor(eval_env, args.logdir)
    # eval_env = RecordVideo(eval_env, f"{args.logdir}/videos", episode_trigger = lambda episode_number: True)
    eval_freq = 5000
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(f"{args.logdir}/models"), n_eval_episodes=3,
                                 log_path=str(f"{args.logdir}/evals"), eval_freq=eval_freq,
                                 deterministic=True, render=False)
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO("MultiInputPolicy", train_env, verbose=1, device=device, tensorboard_log=args.logdir)
    model.learn(total_timesteps=500_000, callback=eval_callback) #, callback=eval_callback)
    #partial_make_base_env = partial(make_base_env, 
    #                                map=args.track,
    #                                fixed_speed=args.fixed_speed,
    #                                random_start =True,)
    #envs = make_vec_env(make_base_env,
    #                    n_envs=args.num_process,
    #                    seed=np.random.randint(pow(2, 31) - 1),
    #                    vec_env_cls=SubprocVecEnv)
    
if __name__ == "__main__":
    train(args)