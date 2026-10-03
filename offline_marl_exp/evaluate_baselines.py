import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from env import DroneSwarmEnv
from bc_masked_actor import MaskedActor
from phase3_online_finetuning import MaskedCritic

class UnmaskedCritic(nn.Module):
    """
    Standard CTDE Critic that receives the full global state.
    For this simulation, we'll assume it takes the concatenated local states 
    (in reality it would take the true unmasked global state from the env).
    """
    def __init__(self, input_dim, n_agents):
        super(UnmaskedCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim * n_agents, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # x is expected to be shape (batch, n_agents, obs_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1) # flatten to global state
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value_head(x)

def run_experiment(config_name, env, actor, critic, is_critic_masked, epochs=20):
    print(f"\n--- Running Experiment: {config_name} ---")
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    
    rewards_history = []
    
    for epoch in range(epochs):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                pred_pos, pred_channels_logits = actor(state_tensor)
                
            pos_actions = pred_pos.numpy()
            channel_actions = torch.argmax(pred_channels_logits, dim=1).numpy()
            
            next_obs, rewards, done, info = env.step(pos_actions, channel_actions)
            episode_reward += np.mean(rewards)
            
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            
            # Critic Evaluation
            if is_critic_masked:
                # Masked critic takes individual local states
                values = critic(state_tensor)
            else:
                # Unmasked critic takes the concatenated global state
                # We add a batch dimension of 1
                global_state = state_tensor.unsqueeze(0)
                # We duplicate the global value for each agent to compute advantage
                global_val = critic(global_state)
                values = global_val.expand(env.n_agents, 1)
            
            advantages = rewards_tensor - values.detach()
            critic_loss = nn.MSELoss()(values, rewards_tensor)
            
            new_pos, new_channels_logits = actor(state_tensor)
            pos_log_prob = -torch.sum((new_pos - torch.tensor(pos_actions))**2, dim=1, keepdim=True)
            actor_loss = -(pos_log_prob * advantages).mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            obs = next_obs
            
        rewards_history.append(episode_reward)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Reward: {episode_reward:.2f}")
            
    return rewards_history

def main():
    input_dim = 16
    n_channels = 3
    n_agents = 3
    epochs = 20
    
    # 1. Load Pre-trained weights for the offline methods
    pretrained_actor = MaskedActor(input_dim=input_dim, n_channels=n_channels)
    try:
        pretrained_actor.load_state_dict(torch.load("masked_actor_policy.pth"))
        print("Loaded pre-trained weights.")
    except:
        print("Pre-trained weights not found! Falling back to random. Run bc_masked_actor.py first.")
        
    results = {}
    env = DroneSwarmEnv(n_agents=n_agents, n_users=10, n_channels=n_channels, k_radius=5.0, max_steps=20)
    
    # --- Baseline A: Pure Online (Random weights, Unmasked CTDE Critic) ---
    actor_A = MaskedActor(input_dim, n_channels) # Random init
    critic_A = UnmaskedCritic(input_dim, n_agents)
    results['Baseline A (Pure Online)'] = run_experiment("Baseline A", env, actor_A, critic_A, False, epochs)
    
    # --- Baseline C: Unmasked Critic (Pre-trained Actor, Unmasked CTDE Critic) ---
    actor_C = copy.deepcopy(pretrained_actor)
    critic_C = UnmaskedCritic(input_dim, n_agents)
    results['Baseline C (Unmasked Critic)'] = run_experiment("Baseline C", env, actor_C, critic_C, False, epochs)
    
    # --- Ours: Asymmetric Masked (Pre-trained Actor, Masked Critic) ---
    actor_Ours = copy.deepcopy(pretrained_actor)
    critic_Ours = MaskedCritic(input_dim)
    results['Ours (Masked Distillation)'] = run_experiment("Ours", env, actor_Ours, critic_Ours, True, epochs)

    # Plotting
    plt.figure(figsize=(10, 6))
    for label, history in results.items():
        plt.plot(history, label=label, marker='o')
        
    plt.title("Online Fine-Tuning: Comparative Evaluation")
    plt.xlabel("Training Epochs")
    plt.ylabel("Episode Reward (Higher is Better)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results_comparison.png")
    print("\nEvaluation complete. Plot saved to results_comparison.png")

if __name__ == "__main__":
    main()
