"""
madt.py -- corrected decentralised Decision Transformer (offline arm; replaces the v1 madt.py).

What was wrong in v1 (review F4, F5) and what is different here:
  * v1 summed [return, state, action] embeddings into ONE token and read the prediction from that token, which
    embedded the very action being predicted (label leakage). Here tokens are INTERLEAVED  [R_1, s_1, a_1, R_2, s_2, a_2, ...]
    under a causal mask, and a_t is read from the output at the s_t token -- a_t itself comes later in the sequence and is
    never visible when predicting it (test: test_gates.py::test_madt_has_no_label_leakage).
  * observations are the env's own k-masked observations (replayed), not a position proxy.
  * the state token is the same permutation-invariant SetEncoder used by the actor, so the student is decentralised
    (it only ever sees o_i^k); each (scenario, agent) is its own sequence, parameters are shared across agents.
Return-to-go is the remaining TEAM return of the expert trajectory; at inference it is set to a target and decremented.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from commenv import CommEnv, EnvConfig, Obs
from models import SetEncoder, dims
from bc import load_dataset


class MADT(nn.Module):
    def __init__(self, cfg: EnvConfig, d=64, layers=3, heads=4, rtg_scale=100.0):
        super().__init__()
        self.cfg, self.d, self.rtg_scale = cfg, d, rtg_scale
        self.enc = SetEncoder(*dims(cfg), d=d)
        self.rtg_emb = nn.Linear(1, d)
        self.act_emb = nn.Linear(2 + cfg.n_channels, d)
        self.time_emb = nn.Embedding(cfg.T + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.pos_head = nn.Linear(d, 2)
        self.ch_head = nn.Linear(d, cfg.n_channels)

    def embed_obs(self, o: Obs):
        return self.enc(o)

    def forward_emb(self, s_emb, a_in, rtg):
        """s_emb [M,L,d]  a_in [M,L,2+C] (actions a_1..a_L; a_L is never used for predicting a_L)  rtg [M,L]
        returns pos [M,L,2], channel logits [M,L,C]"""
        M, L, d = s_emb.shape
        t = torch.arange(L).unsqueeze(0)
        te = self.time_emb(t)
        r = self.rtg_emb(rtg.unsqueeze(-1) / self.rtg_scale) + te
        s = s_emb + te
        a = self.act_emb(a_in) + te
        x = torch.stack([r, s, a], 2).reshape(M, 3 * L, d)                  # interleave R,s,a
        mask = torch.triu(torch.ones(3 * L, 3 * L, dtype=torch.bool), 1)    # causal
        h = self.ln(self.tf(x, mask=mask, is_causal=False))
        hs = h[:, 1::3]                                                     # outputs AT the state tokens
        return self.pos_head(hs), self.ch_head(hs)

    def forward(self, o_seq: Obs, a_in, rtg):
        M, L = a_in.shape[:2]
        flat = Obs(*[x.reshape(M * L, *x.shape[2:]) for x in o_seq])
        s_emb = self.embed_obs(flat).view(M, L, -1)
        return self.forward_emb(s_emb, a_in, rtg)


def action_tokens(a, c, C):
    return torch.cat([a, F.one_hot(c, C).float()], -1)


def build_sequences(seeds, cfg: EnvConfig, actions, channels):
    """Replay the expert; return per-(scenario,agent) sequences: Obs [M,T,..], actions, channels, rtg [M,T]."""
    env = CommEnv(cfg, seeds); obs = env.reset()
    B, N, T = env.B, cfg.n_agents, cfg.T
    O, R = [], []
    for t in range(T):
        O.append(obs.flat())
        obs, r, _, _ = env.step(torch.tensor(actions[:, t]), torch.tensor(channels[:, t]).long())
        R.append(r)
    R = torch.stack(R, 1)                                                    # [B,T]
    rtg = R.flip(1).cumsum(1).flip(1)                                        # remaining team return
    o_seq = Obs(*[torch.stack([getattr(o, f) for o in O], 1) for f in Obs._fields])          # [BN,T,...]
    a = torch.tensor(actions).permute(0, 2, 1, 3).reshape(B * N, T, 2)       # [B,T,N,2] -> [BN,T,2]
    c = torch.tensor(channels).long().permute(0, 2, 1).reshape(B * N, T)
    rtg = rtg.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
    return o_seq, a, c, rtg


def train_madt(cfg: EnvConfig, train_path, val_path, seed=0, epochs=25, bs=48, lr=1e-3, verbose=True):
    torch.manual_seed(seed)
    tr = load_dataset(train_path, cfg); va = load_dataset(val_path, cfg)
    o_tr, a_tr, c_tr, r_tr = build_sequences(tr[0], cfg, tr[1], tr[2])
    o_va, a_va, c_va, r_va = build_sequences(va[0], cfg, va[1], va[2])
    model = MADT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = a_tr.shape[0]
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    C = cfg.n_channels
    t0 = time.time(); hist = []
    def loss_fn(o, a, c, r):
        a_in = action_tokens(a, c, C)
        pos, logits = model(o, a_in, r)
        return F.mse_loss(pos, a) + F.cross_entropy(logits.reshape(-1, C), c.reshape(-1)), F.mse_loss(pos, a).item()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, _ = loss_fn(o_tr.index(idx), a_tr[idx], c_tr[idx], r_tr[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            with torch.no_grad():
                vl = np.mean([loss_fn(o_va.index(torch.arange(i, min(i + 256, o_va.sf.shape[0]))), a_va[i:i + 256], c_va[i:i + 256], r_va[i:i + 256])[1]
                              for i in range(0, o_va.sf.shape[0], 256)])
            hist.append((ep + 1, float(vl)))
            if verbose: print(f"  MADT seed {seed} ep {ep+1:3d} val pos-MSE {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model, hist


class MADTPolicy:
    """Closed-loop, decentralised: every agent keeps its own history of (own obs, own actions) and a return-to-go."""
    def __init__(self, model: MADT, target_return, name="madt"):
        self.m, self.target, self.name = model, float(target_return), name
    @torch.no_grad()
    def act(self, env: CommEnv, obs: Obs):
        cfg, B, N, C = env.cfg, env.B, env.cfg.n_agents, env.cfg.n_channels
        if env.t == 0:
            self.hs, self.ha, self.rtg = [], [], [torch.full((B * N,), self.target)]
        else:
            self.rtg.append(self.rtg[-1] - self.last_r.repeat_interleave(N))
        self.hs.append(self.m.embed_obs(obs.flat()))
        s = torch.stack(self.hs, 1)                                              # [BN,L,d]
        L = s.shape[1]
        a_in = torch.stack(self.ha + [torch.zeros(B * N, 2 + C)], 1)             # placeholder for the not-yet-taken a_L
        pos, logits = self.m.forward_emb(s, a_in, torch.stack(self.rtg, 1))
        a, ch = pos[:, -1], logits[:, -1].argmax(-1)
        self.ha.append(action_tokens(a.clamp(-1.5, 1.5), ch, C))
        return a.view(B, N, 2), ch.view(B, N)
    def observe_reward(self, r):
        self.last_r = r
