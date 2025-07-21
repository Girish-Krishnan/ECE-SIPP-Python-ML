import gymnasium as gym
from stable_baselines3 import DQN


def main():
    env = gym.make('MountainCar-v0')
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save('dqn_mountaincar')
    env.close()

    # Record a short demonstration video
    video_env = gym.wrappers.RecordVideo(
        gym.make('MountainCar-v0', render_mode='rgb_array'),
        video_folder='videos',
        name_prefix='dqn_mountaincar',
        episode_trigger=lambda e: True,
    )
    model = DQN.load('dqn_mountaincar')
    obs, _ = video_env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = video_env.step(action)
    video_env.close()


if __name__ == '__main__':
    main()
