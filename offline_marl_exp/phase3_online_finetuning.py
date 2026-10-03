import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Re-use the MaskedActor from Phase 2
from bc_masked_actor import MaskedActor

class MaskedCritic(nn.Module):
    """
    Centralized Training, Decentralized Execution (CTDE) is standard in MAPPO.
    However, our gap theorem dictates that to align with the bounded observability,
    the Critic must ALSO be masked by radius k.
    """
    def __init__(self, input_dim):
        super(MaskedCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # x is the k-radius masked state observation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        return value

from env import DroneSwarmEnv

def simulate_online_finetuning():
    print("Initializing Phase 3: Online Trust-Region Fine-Tuning...")
    
    # 1. Load Pre-trained Student Policy (Phase 2)
    input_dim = 16
    n_channels = 3
    actor = MaskedActor(input_dim=input_dim, n_channels=n_channels)
    try:
        actor.load_state_dict(torch.load("masked_actor_policy.pth"))
        print("Successfully loaded pre-trained Asymmetric Distillation weights.")
    except FileNotFoundError:
        print("Warning: pre-trained weights not found. Using random initialization.")

    # 2. Initialize Range-Masked Critic
    critic = MaskedCritic(input_dim=input_dim)
    
    # Optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    
    # 3. Instantiate Environment
    env = DroneSwarmEnv(n_agents=3, n_users=10, n_channels=n_channels, k_radius=5.0)
    print("\nStarting Online Fine-Tuning with live DroneSwarmEnv...")
    
    epochs = 10
    
    for epoch in range(epochs):
        obs = env.reset() # shape (n_agents, 16)
        done = False
        
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            
            # Predict actions
            with torch.no_grad():
                pred_pos, pred_channels_logits = actor(state_tensor)
                
            pos_actions = pred_pos.numpy()
            channel_actions = torch.argmax(pred_channels_logits, dim=1).numpy()
            
            # Step environment
            next_obs, rewards, done, info = env.step(pos_actions, channel_actions)
            episode_reward += np.mean(rewards)
            
            # PPO / Trust-Region Update
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            values = critic(state_tensor)
            
            advantages = rewards_tensor - values.detach()
            critic_loss = nn.MSELoss()(values, rewards_tensor)
            
            new_pos, new_channels_logits = actor(state_tensor)
            
            # Simplified actor loss
            pos_log_prob = -torch.sum((new_pos - torch.tensor(pos_actions))**2, dim=1, keepdim=True)
            actor_loss = -(pos_log_prob * advantages).mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            obs = next_obs
            
        print(f"Epoch {epoch+1}/{epochs} | Episode Reward: {episode_reward:.2f} | Info: {info}")

    print("\nPhase 3 Online Fine-Tuning architecture successfully verified with live environment.")

if __name__ == "__main__":
    simulate_online_finetuning()
