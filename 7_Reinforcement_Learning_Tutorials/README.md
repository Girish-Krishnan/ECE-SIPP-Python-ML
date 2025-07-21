# Reinforcement Learning Tutorials

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the Jupyter Notebook in your browser, in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/7_Reinforcement_Learning_Tutorials/reinforcement_learning_tutorials.ipynb)

This option does not require you to install anything locally. The notebook contains step-by-step explanations and runnable code cells.
It now also records short demo videos after training each agent.

## Option 2: Python Scripts

Install the required packages with pip:

```bash
pip install stable-baselines3 gymnasium
```

For the Humanoid example you will also need the `mujoco` physics engine:

```bash
pip install mujoco
```

### 0. Train PPO on CartPole
`python 00_train_cartpole.py`

Trains a PPO agent on the CartPole-v1 environment and saves `ppo_cartpole.zip`.
After training, the script runs one episode with the trained policy and
records a short video to the `videos/` directory.

### 1. Train DQN on MountainCar
`python 01_train_mountaincar.py`

Trains a DQN agent on the MountainCar-v0 environment and saves `dqn_mountaincar.zip`.
A short inference run is also recorded to `videos/`.

### 2. Humanoid example with PPO
`python 02_humanoid_ppo.py`

Runs PPO on the Humanoid-v4 environment for a short training run (requires Mujoco).
A demonstration video is written to `videos/`.
