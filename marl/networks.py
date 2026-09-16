"""Small MLPs (actor, centralized critic) built on the autodiff engine."""

from __future__ import annotations
import numpy as np
from .autodiff import Tensor


def _init_linear(in_dim, out_dim, rng):
    limit = np.sqrt(6.0 / (in_dim + out_dim))  # Xavier/Glorot uniform
    W = Tensor(rng.uniform(-limit, limit, size=(in_dim, out_dim)))
    b = Tensor(np.zeros(out_dim))
    return W, b


class MLP:
    """Two hidden ReLU layers + a linear output head."""

    def __init__(self, in_dim, hidden_dim, out_dim, rng):
        self.W1, self.b1 = _init_linear(in_dim, hidden_dim, rng)
        self.W2, self.b2 = _init_linear(hidden_dim, hidden_dim, rng)
        self.W3, self.b3 = _init_linear(hidden_dim, out_dim, rng)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: Tensor) -> Tensor:
        h1 = x.matmul(self.W1) + self.b1
        h1 = h1.relu()
        h2 = h1.matmul(self.W2) + self.b2
        h2 = h2.relu()
        out = h2.matmul(self.W3) + self.b3
        return out


class CategoricalActor:
    """Softmax policy over a small discrete action set."""

    def __init__(self, obs_dim, n_actions, hidden_dim, rng):
        self.net = MLP(obs_dim, hidden_dim, n_actions, rng)

    def params(self):
        return self.net.params()

    def forward_probs(self, obs: Tensor) -> Tensor:
        logits = self.net.forward(obs)
        return logits.softmax(axis=-1)

    def act(self, obs_np, rng):
        """Sample an action for a single observation (numpy in, numpy out)."""
        obs_t = Tensor(obs_np[None, :], requires_grad=False)
        probs = self.forward_probs(obs_t).data[0]
        probs = probs / probs.sum()
        action = rng.choice(len(probs), p=probs)
        logprob = np.log(probs[action] + 1e-8)
        return int(action), float(logprob), probs


class CentralizedCritic:
    """Value function over the concatenated (global) observation."""

    def __init__(self, global_state_dim, hidden_dim, rng):
        self.net = MLP(global_state_dim, hidden_dim, 1, rng)

    def params(self):
        return self.net.params()

    def forward_value(self, state: Tensor) -> Tensor:
        return self.net.forward(state)  # shape (batch, 1)

    def value(self, state_np):
        state_t = Tensor(state_np[None, :], requires_grad=False)
        return float(self.forward_value(state_t).data[0, 0])
