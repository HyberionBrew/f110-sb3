# MIT License

# Copyright (c) 2021 Eoin Gogarty, Charlie Maguire and Manus McAuliffe (Formula Trintiy Autonomous)

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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Sample python script to show the process of choosing a Stable Baselines model,
training it with a chosen policy, and then evaluating the trained model on the
environment while visualising it
"""
#if getting the module not found error run;
#pip install --user -e gym/
#in the f1tenth_gym folder

import gymnasium as gym
import time
import numpy as np
import torch
import os
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from code.wrappers import F110_Wrapped, ThrottleMaxSpeedReward, RandomStartPosition # , FixSpeedControl
from code.reward_wrappers import ProgressReward
from code.manus_callbacks import SaveOnBestTrainingRewardCallback
from code.schedulers import linear_schedule

from base_env import make_base_env

from pyglet.gl import GL_POINTS

TRAIN_DIRECTORY = "./train"
TRAIN_STEPS = 1.5 * np.power(10, 6)    # for reference, it takes about one sec per 500 steps
SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 10)
MIN_EVAL_EPISODES = 100
NUM_PROCESS = 1
MAP_PATH = "./f1tenth_racetracks/maps-felix_test2/infsaal"
MAP_EXTENSION = ".pgm"

class Waypoints(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.drawn_waypoints = []

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.xs, self.ys)).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

def main():

    #       #
    # TRAIN #
    #       #



    drawn_waypoints = []


    # env = make_base_env(fixed_speed=True)
    env = gym.make("f110_gym:f110-v0",
                        config = dict(map="Infsaal",
                        num_agents=1),
                        render_mode="human")
    # env = F110_Wrapped(env)
    # print(env.reset()) #options=dict(poses=[np.asarray([0.0,0.0]),1])))
    # env
    #action = np.array([0.0,0.0])
    #print(env.step(action))
    # exit()
    #print(env.track)
    
    # exit()
    x = env.track.centerline.xs 
    y = env.track.centerline.ys
    waypoints = Waypoints(x,y)

    def render_callback(env_renderer):
        # custom extra drawing function
        waypoints.render_waypoints(env_renderer)     
    
    # env.add_render_callback(render_callback)
    """
    episode=0
    while True: #episode < MIN_EVAL_EPISODES:
    
        episode += 1
        obs, _ = env.reset()
        done = False
        
        while not done:
            # print(obs)
            action = np.array([0.0,0.0]) #, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                print("Lap done")
            print("R:", reward)
        
            env.render()
    """
    # exit(0)
    # vectorise environment (parallelise)
    # wrap_env = env
    envs = make_vec_env(make_base_env,
                        n_envs=NUM_PROCESS,
                        seed=np.random.randint(pow(2, 31) - 1),
                        vec_env_cls=SubprocVecEnv)

    # choose RL model and policy here
    """eval_env = gym.make("f110_gym:f110-v0",map=MAP_PATH,map_ext=MAP_EXTENSION,num_agents=1)
    eval_env = F110_Wrapped(eval_env)
    eval_env = RandomF1TenthMap(eval_env, 500)
    eval_env.seed(np.random.randint(pow(2, 31) - 1))"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #RuntimeError: CUDA error: out of memory whenever I use gpu
    print(f"Using device: {device}")
    # load latest model if it exists format is ppo-f110-<timestamp>
    model = PPO("MlpPolicy", envs,  learning_rate=linear_schedule(0.0003), gamma=0.99, gae_lambda=0.95, verbose=1, device=device)
    eval_callback = EvalCallback(envs, best_model_save_path='./train_test/',
                             log_path='./train_test/', eval_freq=5000,
                             deterministic=True, render=False)

    # train model and record time taken
    start_time = time.time()
    model.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback)
    print(f"Training time {time.time() - start_time:.2f}s")
    print("Training cycle complete.")

    # save model with unique timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model.save(f"./train/ppo-f110-{timestamp}")

    #          #
    # EVALUATE #
    #          #

    # create evaluation environment (same as train environment in this case)
    """
    eval_env = gym.make("f110_gym:f110-v0",
                        map=MAP_PATH,
                        map_ext=MAP_EXTENSION,
                        num_agents=1)

    # wrap evaluation environment
    eval_env = F110_Wrapped(eval_env)
    eval_env = RandomF1TenthMap(eval_env, 500)
    eval_env.seed(np.random.randint(pow(2, 31) - 1))
    model = model.load("./train_test/best_model")

    # simulate a few episodes and render them, ctrl-c to cancel an episode
    episode = 0
    while episode < MIN_EVAL_EPISODES:
        try:
            episode += 1
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = eval_env.step(action)
                eval_env.render()
        except KeyboardInterrupt:
            pass
    """

# necessary for Python multi-processing
if __name__ == "__main__":
    main()