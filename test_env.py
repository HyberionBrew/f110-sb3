import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import base_env
def main():
    env = base_env.make_base_env()
    check_env(env, warn=True, skip_render_check=False)
if __name__ == '__main__':
    main()