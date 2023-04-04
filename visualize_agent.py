import numpy as np
import torch
from unityagents import UnityEnvironment
import time
from agent import Agent

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize(actor_filepath, critic_filepath, env_exe_path, n_steps=100, sleep_between_frames=0):
    # Load Environment
    env = UnityEnvironment(file_name=env_exe_path)
    brain_name = env.brain_names[0]

    print('Device', DEVICE)
    print('*' * 50)

    # Load trained agent
    agent = Agent(env, gamma=0.99, tau=0.001,
                  batch_size=128, buffer_size=int(1e6),
                  lr_actor=0.001, lr_critic=0.0001,
                  local_update_freq=1, target_update_freq=10, device=DEVICE)

    agent.load(actor_filepath, critic_filepath)

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    agent.noise.reset()  # reset the noise
    rewards = 0

    for _ in range(n_steps):
        action = agent.select_action(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        env_info.local_done[0]
        state = next_state
        rewards += reward
        time.sleep(sleep_between_frames)

    print('*' * 50)
    print(f"Total Reward: {rewards:.2f}")

    env.close()


def visualize_random(env_exe_path, n_steps=100, sleep_between_frames=0):
    # Load Environment
    env = UnityEnvironment(file_name=env_exe_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    n_actions = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)
    rewards = 0

    for _ in range(n_steps):
        action = np.random.randn(n_agents, n_actions)
        action = np.clip(action, -1, 1)
        env_info = env.step(action)[brain_name]
        reward = env_info.rewards[0]
        rewards += reward
        time.sleep(sleep_between_frames)

    print('*' * 50)
    print(f"Total Reward: {rewards:.2f}")

    env.close()
