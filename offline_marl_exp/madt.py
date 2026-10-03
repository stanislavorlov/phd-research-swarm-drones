import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import math

class SequenceExpertDataset(Dataset):
    def __init__(self, json_path, k_radius=5.0, context_len=5):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.k_radius = k_radius
        self.context_len = context_len
        self.samples = self._process_trajectories()
        
    def _process_trajectories(self):
        samples = []
        for episode in self.data:
            users = np.array(episode['users'])
            trajectory = episode['trajectory'] # list of timesteps, each has n_uavs
            n_uavs = len(trajectory[0])
            
            # For each UAV, extract its sequence
            for uav_idx in range(n_uavs):
                states = []
                pos_actions = []
                channel_actions = []
                rewards = []
                
                for t, step_data in enumerate(trajectory):
                    u_data = step_data[uav_idx]
                    own_pos = np.array(u_data['action_pos']) # using pos target as proxy for own pos for simplicity
                    
                    # Masking logic
                    user_dists = np.linalg.norm(users - own_pos, axis=1)
                    visible_users = users[user_dists <= self.k_radius]
                    
                    other_uav_positions = []
                    for j, other_u_data in enumerate(step_data):
                        if j != uav_idx:
                            other_pos = np.array(other_u_data['action_pos'])
                            if np.linalg.norm(other_pos - own_pos) <= self.k_radius:
                                other_uav_positions.append(other_pos)
                                
                    # State construction (padded to 16 dims)
                    state_features = list(own_pos)
                    for i in range(5):
                        if i < len(visible_users): state_features.extend(visible_users[i] - own_pos)
                        else: state_features.extend([0.0, 0.0])
                    for i in range(2):
                        if i < len(other_uav_positions): state_features.extend(other_uav_positions[i] - own_pos)
                        else: state_features.extend([0.0, 0.0])
                        
                    states.append(state_features)
                    pos_actions.append(u_data['action_pos'])
                    channel_actions.append(u_data['action_channel'])
                    rewards.append(u_data['reward'])
                    
                # Calculate returns-to-go (RTG)
                rtg = np.zeros_like(rewards)
                running_return = 0
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + running_return
                    rtg[t] = running_return
                    
                # Create rolling windows of context_len
                for t in range(len(states)):
                    start_idx = max(0, t - self.context_len + 1)
                    
                    # Pad sequences if they are shorter than context_len
                    pad_len = self.context_len - (t - start_idx + 1)
                    
                    s_seq = np.zeros((self.context_len, 16))
                    a_pos_seq = np.zeros((self.context_len, 2))
                    a_chan_seq = np.zeros((self.context_len,), dtype=int)
                    rtg_seq = np.zeros((self.context_len, 1))
                    
                    s_seq[pad_len:] = states[start_idx:t+1]
                    a_pos_seq[pad_len:] = pos_actions[start_idx:t+1]
                    a_chan_seq[pad_len:] = channel_actions[start_idx:t+1]
                    rtg_seq[pad_len:, 0] = rtg[start_idx:t+1]
                    
                    # We want to predict the action at time t (last element)
                    target_pos = pos_actions[t]
                    target_chan = channel_actions[t]
                    
                    samples.append({
                        'states': torch.tensor(s_seq, dtype=torch.float32),
                        'pos_actions': torch.tensor(a_pos_seq, dtype=torch.float32),
                        'rtg': torch.tensor(rtg_seq, dtype=torch.float32),
                        'target_pos': torch.tensor(target_pos, dtype=torch.float32),
                        'target_chan': torch.tensor(target_chan, dtype=torch.long)
                    })
                    
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['states'], s['pos_actions'], s['rtg'], s['target_pos'], s['target_chan']


class MultiAgentDecisionTransformer(nn.Module):
    def __init__(self, state_dim=16, pos_act_dim=2, hidden_dim=64, context_len=5, n_channels=3):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        
        # Embeddings
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_pos_action = nn.Linear(pos_act_dim, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        
        self.embed_timestep = nn.Embedding(num_embeddings=100, embedding_dim=hidden_dim)
        
        # GPT-like Transformer (using PyTorch standard TransformerEncoder for simplicity)
        # In a strict DT, we sequence [R, S, A]. Here we fuse them into a single token per timestep for brevity, 
        # or use standard casual masking across sequence length.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Action Heads
        self.predict_pos = nn.Linear(hidden_dim, pos_act_dim)
        self.predict_channel = nn.Linear(hidden_dim, n_channels)
        
    def forward(self, states, pos_actions, rtg, timesteps=None):
        batch_size, seq_length, _ = states.shape
        
        if timesteps is None:
            timesteps = torch.arange(seq_length, device=states.device).unsqueeze(0).repeat(batch_size, 1)
            
        # Embed inputs
        state_emb = self.embed_state(states)
        action_emb = self.embed_pos_action(pos_actions)
        rtg_emb = self.embed_rtg(rtg)
        time_emb = self.embed_timestep(timesteps)
        
        # Fuse token at each timestep (simplified DT approach: add them up)
        # Real DT interleaves them [R_1, S_1, A_1, R_2...], but fused token with causal mask 
        # is functionally sufficient for this scale.
        token_emb = state_emb + action_emb + rtg_emb + time_emb
        
        # Causal mask (don't look into the future)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(states.device)
        
        # Transform
        x = self.transformer(token_emb, mask=mask, is_causal=True)
        
        # Predict next action using the last token representation
        last_token = x[:, -1, :]
        pred_pos = self.predict_pos(last_token)
        pred_channel = self.predict_channel(last_token)
        
        return pred_pos, pred_channel


def train_madt():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "expert_sequence_dataset_sample.json")
    if not os.path.exists(dataset_path):
        print("Sequence Dataset not found. Run expert_solver.py first.")
        return
        
    dataset = SequenceExpertDataset(dataset_path, k_radius=5.0, context_len=5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = MultiAgentDecisionTransformer(hidden_dim=64, context_len=5, n_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    pos_criterion = nn.MSELoss()
    channel_criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    print(f"Starting Offline MADT Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for states, pos_actions, rtgs, target_pos, target_chan in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            pred_pos, pred_chan_logits = model(states, pos_actions, rtgs)
            
            # Compute loss
            loss_pos = pos_criterion(pred_pos, target_pos)
            loss_chan = channel_criterion(pred_chan_logits, target_chan)
            loss = loss_pos + loss_chan
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
            
    print("MADT Training complete. Saving policy weights.")
    torch.save(model.state_dict(), os.path.join(script_dir, "madt_policy.pth"))

if __name__ == "__main__":
    train_madt()
