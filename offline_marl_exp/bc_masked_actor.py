import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==========================================
# RESEARCH TITLE: 
# Range-Bounded Asymmetric Distillation for Multi-Agent Trajectory and Resource Optimization
# (Offline-to-Online MARL with k-radius masked critics)
# ==========================================

class ExpertDataset(Dataset):
    def __init__(self, json_path, k_radius=5.0):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.k_radius = k_radius

    def __len__(self):
        # Each scenario contains multiple UAVs, we treat each UAV's decision as a sample
        return len(self.data) * len(self.data[0]['optimal_uavs'])

    def __getitem__(self, idx):
        scenario_idx = idx // len(self.data[0]['optimal_uavs'])
        uav_idx = idx % len(self.data[0]['optimal_uavs'])
        
        scenario = self.data[scenario_idx]
        users = np.array(scenario['users'])
        uavs = scenario['optimal_uavs']
        
        own_uav = uavs[uav_idx]
        own_pos = np.array(own_uav['pos'])
        
        # MASKING: Only observe users within k_radius
        user_dists = np.linalg.norm(users - own_pos, axis=1)
        visible_users = users[user_dists <= self.k_radius]
        
        # MASKING: Only observe other UAVs within k_radius
        other_uav_positions = []
        for j, u in enumerate(uavs):
            if j != uav_idx:
                other_pos = np.array(u['pos'])
                if np.linalg.norm(other_pos - own_pos) <= self.k_radius:
                    other_uav_positions.append(other_pos)
                    
        # Fixed size state representation (padding with zeros if no entities observed)
        # For simplicity in this script, we'll use a fixed-size local density grid 
        # or just pad the nearest N visible entities. 
        # Let's pad up to 5 users and 2 other UAVs for the MLP input.
        max_visible_users = 5
        max_visible_uavs = 2
        
        state_features = []
        state_features.extend(own_pos) # 2 dims
        
        # Add visible users (relative positions)
        for i in range(max_visible_users):
            if i < len(visible_users):
                state_features.extend(visible_users[i] - own_pos)
            else:
                state_features.extend([0.0, 0.0]) # padded
                
        # Add visible UAVs (relative positions)
        for i in range(max_visible_uavs):
            if i < len(other_uav_positions):
                state_features.extend(other_uav_positions[i] - own_pos)
            else:
                state_features.extend([0.0, 0.0]) # padded
                
        # Targets
        target_pos = torch.tensor(own_pos, dtype=torch.float32)
        target_channel = torch.tensor(own_uav['channel'], dtype=torch.long)
        
        return torch.tensor(state_features, dtype=torch.float32), target_pos, target_channel

class MaskedActor(nn.Module):
    def __init__(self, input_dim, n_channels=3):
        super(MaskedActor, self).__init__()
        # Shared feature extractor
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Continuous action head (Trajectory/Position)
        # Note: In a real RL setup, this would output mean & variance for a Gaussian
        self.pos_head = nn.Linear(64, 2)
        
        # Discrete action head (Resource/Channel Allocation)
        self.channel_head = nn.Linear(64, n_channels)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        pos = self.pos_head(x)
        channel_logits = self.channel_head(x)
        return pos, channel_logits

def train_behavioral_cloning():
    dataset_path = "data/expert_dataset_sample.json"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please run expert_solver.py first.")
        return
        
    dataset = ExpertDataset(dataset_path, k_radius=5.0)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # input_dim = 2 (own) + 2*5 (users) + 2*2 (uavs) = 16
    model = MaskedActor(input_dim=16, n_channels=3)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pos_criterion = nn.MSELoss()
    channel_criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    print(f"Starting Asymmetric Distillation (Behavioral Cloning) for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for state, target_pos, target_channel in dataloader:
            optimizer.zero_grad()
            
            pred_pos, pred_channel_logits = model(state)
            
            # Hybrid Loss: MSE for trajectory, Cross-Entropy for resource assignment
            loss_pos = pos_criterion(pred_pos, target_pos)
            loss_channel = channel_criterion(pred_channel_logits, target_channel)
            
            loss = loss_pos + loss_channel
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
            
    print("Training complete. Saving student policy.")
    torch.save(model.state_dict(), "masked_actor_policy.pth")

if __name__ == "__main__":
    train_behavioral_cloning()
