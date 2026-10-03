import numpy as np

class DroneSwarmEnv:
    def __init__(
        self,
        n_agents: int = 3,
        n_users: int = 10,
        n_channels: int = 3,
        world_size: float = 10.0,
        max_steps: int = 20,
        k_radius: float = 5.0,
        seed: int = None,
    ):
        self.n_agents = n_agents
        self.n_users = n_users
        self.n_channels = n_channels
        self.world_size = world_size
        self.max_steps = max_steps
        self.k_radius = k_radius
        self.rng = np.random.default_rng(seed)

        # Observation dimension: own pos (2) + 5 users max (5*2) + 2 other UAVs max (2*2) = 16
        self.obs_dim = 16
        
        self.positions = None
        self.channels = None
        self.users = None
        self.t = 0

    def reset(self):
        # Random initial positions and channels
        self.positions = self.rng.uniform(0, self.world_size, size=(self.n_agents, 2))
        self.channels = self.rng.integers(0, self.n_channels, size=(self.n_agents,))
        self.users = self.rng.uniform(0, self.world_size, size=(self.n_users, 2))
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            own_pos = self.positions[i]
            
            # Masking: only see users within k_radius
            user_dists = np.linalg.norm(self.users - own_pos, axis=1)
            visible_users = self.users[user_dists <= self.k_radius]
            
            # Masking: only see other UAVs within k_radius
            other_uavs = []
            for j in range(self.n_agents):
                if i != j:
                    if np.linalg.norm(self.positions[j] - own_pos) <= self.k_radius:
                        other_uavs.append(self.positions[j])
                        
            # Build fixed-size state vector (padded)
            state_features = []
            state_features.extend(own_pos)
            
            # Add up to 5 visible users (relative positions)
            for k in range(5):
                if k < len(visible_users):
                    state_features.extend(visible_users[k] - own_pos)
                else:
                    state_features.extend([0.0, 0.0])
                    
            # Add up to 2 visible UAVs (relative positions)
            for k in range(2):
                if k < len(other_uavs):
                    state_features.extend(other_uavs[k] - own_pos)
                else:
                    state_features.extend([0.0, 0.0])
                    
            obs[i] = np.array(state_features)
        return obs

    def step(self, pos_actions, channel_actions):
        """
        pos_actions: (n_agents, 2) array of target positions
        channel_actions: (n_agents,) array of channel indices
        """
        # Continuous movement: move towards target up to a max step size (e.g. 0.5)
        moves = pos_actions - self.positions
        dists = np.linalg.norm(moves, axis=1, keepdims=True)
        # Cap movement speed
        moves = np.where(dists > 0.5, (moves / dists) * 0.5, moves)
        
        self.positions = np.clip(self.positions + moves, 0, self.world_size)
        self.channels = np.clip(channel_actions, 0, self.n_channels - 1)
        self.t += 1

        # Calculate reward: (negative objective function from expert solver)
        total_coverage_cost = 0
        for i in range(self.n_users):
            user_dists = np.linalg.norm(self.positions - self.users[i], axis=1)
            total_coverage_cost += np.min(user_dists)
            
        interference_penalty = 0
        for j1 in range(self.n_agents):
            for j2 in range(j1 + 1, self.n_agents):
                if self.channels[j1] == self.channels[j2]:
                    dist_between = np.linalg.norm(self.positions[j1] - self.positions[j2])
                    if dist_between < 4.0:
                        interference_penalty += (4.0 - dist_between) * 10.0
                        
        # Global reward (shared by all agents)
        reward = -(total_coverage_cost + interference_penalty)
        rewards = np.full(self.n_agents, reward, dtype=np.float32)

        done = self.t >= self.max_steps
        info = {
            "coverage_cost": total_coverage_cost,
            "interference_penalty": interference_penalty
        }
        return self._get_obs(), rewards, done, info
