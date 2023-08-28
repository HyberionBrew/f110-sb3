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
from stable_baselines3.common.callbacks import BaseCallback
#TODO! normalization is only applied process wise, not across processes (i think)

parser = argparse.ArgumentParser(description='Train a model on the F1Tenth Gym environment')
parser.add_argument('--logdir', type=str, default='logs', help='Logging directory')
parser.add_argument('--track', type=str, default='Infsaal', help='Track to train on')
parser.add_argument('--fixed_speed', type=float, default=None, help='Fixing the speed to the provided value')
parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes')
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

class RewardLoggerCallback(BaseCallback):
    def __init__(self,env, verbose=0):
        self.env = env
    def _on_step(self) -> bool:
        # Access the environment using self.locals['env']
        # Example: logging the current episode reward
        # (this assumes the reward is stored in your custom Gym environment as 'current_reward')
        reward = self.env.all_rewards
        print(reward)
        self.logger.record('train/custom_reward', reward)
        
        # Return True to continue training, False to stop
        return True
    
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.all_rewards
        self.logger.record("TD", value[0])
        return True


def train(args):
    
    reward_config = {
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

    reward_config_eval = {
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




    save_parameters_to_file(args.logdir, args)
    # create logdir
    #train_env = make_base_env(map= args.track,
    #                fixed_speed=args.fixed_speed,
    #                random_start =True,
    #                reward_config = reward_config)
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
    eval_freq = 10_000
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(f"{args.logdir}/models"), n_eval_episodes=5,
                                 log_path=str(f"{args.logdir}/evals"), eval_freq=eval_freq,
                                 deterministic=True, render=False)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO("MultiInputPolicy", train_envs, verbose=1, device=device, tensorboard_log=args.logdir)
    # model.learn(total_timesteps=500_000, callback=[eval_callback,RewardLoggerCallback(train_envs)], progress_bar=True) #, callback=eval_callback)
    model.learn(total_timesteps=500_000, callback=[eval_callback], progress_bar=True) #, callback=eval_callback)
if __name__ == "__main__":
    train(args)