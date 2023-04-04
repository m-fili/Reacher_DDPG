import numpy as np


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process.
    dX_t = θ(μ - X_t) dt + σ dW_t
    """

    def __init__(self, size, mu=0., theta=0.1, sigma=0.2, random_seed=72):
        """
        Initialize parameters and noise process.
        :param size: Dimension of the noise
        :param mu: Mean of the noise
        :param theta: Rate of mean reversion
        :param sigma: Standard deviation of the noise
        :param random_seed: Random seed
        """
        self.seed = np.random.seed(random_seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.noise = self.mu.copy()

    def generate(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.noise
        dw = np.array([np.random.normal() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * dw
        self.noise = x + dx
        return self.noise

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.noise = self.mu.copy()
