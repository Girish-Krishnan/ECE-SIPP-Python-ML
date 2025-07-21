import gymnasium as gym
from stable_baselines3 import DQN


def main():
    env = gym.make('MountainCar-v0')
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save('dqn_mountaincar')
    env.close()


if __name__ == '__main__':
    main()
