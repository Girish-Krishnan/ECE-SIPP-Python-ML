import gymnasium as gym
from stable_baselines3 import PPO


def main():
    env = gym.make('Humanoid-v4')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save('ppo_humanoid')
    env.close()

    # Record a short demonstration video
    video_env = gym.wrappers.RecordVideo(
        gym.make('Humanoid-v4', render_mode='rgb_array'),
        video_folder='videos',
        name_prefix='ppo_humanoid',
        episode_trigger=lambda e: True,
    )
    model = PPO.load('ppo_humanoid')
    obs, _ = video_env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = video_env.step(action)
    video_env.close()


if __name__ == '__main__':
    main()
