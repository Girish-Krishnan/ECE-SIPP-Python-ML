import gymnasium as gym
from stable_baselines3 import PPO


def main():
    env = gym.make('Humanoid-v4')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save('ppo_humanoid')
    env.close()


if __name__ == '__main__':
    main()
