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
def evaluate(args):
    eval_env = make_base_env(map= args.track,
                    fixed_speed=args.fixed_speed,
                    random_start =True,)
    eval_env = TimeLimit(eval_env, max_episode_steps=500)
    
    model = PPO.load(args.model_path)
    episode = 0
    while episode < 2000:
        episode += 1
        obs, _ = eval_env.reset()
        done = False
        rew = 0
        steps = 0
        rewards = []
        while not done:
            # print(obs)
            steps += 1
            action, _ = model.predict(obs)
            # print(action)
            # print(action.shape)
            obs, reward, done, truncated, _ = eval_env.step(action)
            print(reward)
            rewards.append(reward)
            rew += reward * 0.99
            if truncated:
                print(steps)
                print("Truncated")

            if done or truncated:
                print(steps)
                print("Lap done")
                print("R:", rew)
                plt.plot(rewards)
                plt.show()
            eval_env.render()

if __name__ == "__main__":
    evaluate(args)