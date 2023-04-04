import numpy as np
from collections import deque


def ddpg(env, agent, n_episodes=1000, print_every=100):
    """Deep Deterministic Policy Gradient."""

    scores = []
    scores_deque = deque(maxlen=print_every)
    time_step = 1

    # get the default brain
    brain_name = env.brain_names[0]


    for i_episode in range(1, n_episodes + 1):

        rewards = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.noise.reset()     # reset the noise
        done = False

        while not done:
            action = agent.select_action(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done, time_step)
            state = next_state
            rewards += reward
            time_step += 1

            if done:
                break

        scores.append(rewards)
        scores_deque.append(rewards)

        print(f'\rEpisode {i_episode}\t Score: {rewards:.2f} ({np.mean(scores_deque):.2f})', end='\r')
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode:4}\tAverage Score: {np.mean(scores_deque):.2f}{"":10}')

    return scores
