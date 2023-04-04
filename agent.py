import numpy as np
import torch
import torch.nn.functional as F
from models import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise


class Agent:

    def __init__(self, env, gamma=0.99, tau=1e-3, batch_size=128, buffer_size=512,
                 lr_actor=1e-4, lr_critic=1e-3, local_update_freq=1,
                 target_update_freq=1, device=None, random_seed=72):
        """
        Initialize an Agent object.
        :param env: Environment
        :param gamma: Discount factor
        :batch_size: number of experiences to sample from the replay buffer
        :buffer_size: maximum size of the replay buffer
        :param tau: Mixing factor for soft update
        :param lr_actor: Learning rate for actor
        :param lr_critic: Learning rate for critic
        :param local_update_freq: Update frequency for local networks
        :param target_update_freq: Update frequency for target networks
        :param random_seed: random seed
        """

        # general
        # brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        self.n_states = brain.vector_observation_space_size
        self.n_actions = brain.vector_action_space_size
        self.seed = random_seed
        self.action_lb = -1
        self.action_ub = +1
        self.tau = tau
        self.gamma = gamma
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # actor
        self.actor_local = Actor(self.n_states, self.n_actions, self.seed).to(self.device)
        self.actor_target = Actor(self.n_states, self.n_actions, self.seed).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # critic
        self.critic_local = Critic(self.n_states, self.n_actions, self.seed).to(self.device)
        self.critic_target = Critic(self.n_states, self.n_actions, self.seed).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic)

        # memory
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # update frequency
        self.local_update_freq = local_update_freq
        self.target_update_freq = target_update_freq

        # noise
        self.noise = OrnsteinUhlenbeckNoise(size=self.n_actions, mu=0., theta=0.1,
                                            sigma=0.2, random_seed=self.seed)



    def select_action(self, state, add_noise=True):
        """
        Select an action from the input state.
        :param state:
        :param add_noise:
        :return:
        """
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if add_noise:
            action += self.noise.generate()
        return np.clip(action, self.action_lb, self.action_ub)

    def step(self, state, action, reward, next_state, done, time_step):
        # Save experience in replay memory
        self.memory.add_experience(state, action, reward, next_state, done)

        if self.memory.ready_to_learn():
            if time_step % self.local_update_freq == 0:
                self.update_local_networks()  # update the local network

            if time_step % self.target_update_freq == 0:
                self.update_target_networks()  # soft update the target network towards the actual networks

    def update_local_networks(self):
        """
        Update the local networks (actor and critic) using the experiences sampled from memory.
        """
        # sample a batch of experiences from memory
        states, actions, rewards, next_states, dones = self.memory.sample_experience()
        # convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        # update actor and critic networks
        self._update_actor(states)
        self._update_critic(states, actions, rewards, next_states, dones)

    def update_target_networks(self):
        """Soft update model parameters of the target network towards the local network."""
        self._soft_update(self.actor_local, self.actor_target)
        self._soft_update(self.critic_local, self.critic_target)

    def _update_actor(self, states):
        """
        Update the actor network using Q(s, mu(s)) as the loss function,
        where mu(s) is the output of the actor network (predicted action).
        """
        predicted_actions = self.actor_local(states)
        loss = - self.critic_local(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def _update_critic(self, states, actions, rewards, next_states, dones):
        """ we want to minimize the loss function L = (Q(s,a) - y)**2
        where y = r + gamma * Q(s', mu(s')) * (1 - done)
        """
        # get the predicted next actions
        predicted_next_actions = self.actor_target(next_states)
        # get the predicted Q values for the next states and actions
        Q_targets_next = self.critic_target(next_states, predicted_next_actions)
        # compute the target Q values
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # get the predicted Q values for the current states and actions
        Q_expected = self.critic_local(states, actions)
        # compute the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # minimize the loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _soft_update(self, local_network, target_network):
        """
        Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_network: The model that is being trained
        :param target_network: The model that is being updated
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, actor_filepath, critic_filepath):
        torch.save(self.actor_local.state_dict(), actor_filepath)
        print("Actor successfully saved to {}".format(actor_filepath))

        torch.save(self.critic_local.state_dict(), critic_filepath)
        print("Critic successfully saved to {}".format(critic_filepath))

    def load(self, actor_filepath, critic_filepath):
        self.actor_local.load_state_dict(torch.load(actor_filepath))
        print("Actor successfully loaded from {}".format(actor_filepath))

        self.critic_local.load_state_dict(torch.load(critic_filepath))
        print("Critic successfully loaded from {}".format(critic_filepath))
