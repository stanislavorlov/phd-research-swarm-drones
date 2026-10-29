"""
iql.py -- Implicit Q-Learning on the planner dataset (paper 2, offline arm).

    python3 iql.py --name I_weak1 --seed 0 --train_data data/expert_train_weak1.pt \
        --val_data data/expert_val_weak1.pt --k 6 --k_a -1 --bcast 6 --head mdn --d 128 \
        --mdn_k 5 --chunk 4 --epochs0 300
    python3 iql.py --smoke                      # seconds, end to end, results meaningless

Same contract as student.py: writes results2/<name>/seed<s>/best.pt via bc2.save_student, so final_eval.py
and online_test_eval.py pick it up with no changes.

Why IQL rather than MA-CQL: no per-agent conservatism coefficient to tune, and its policy-extraction step IS
weighted behaviour cloning, so it reuses the loss this project already trusts.

  V:  expectile regression towards Q_target(o,a)          L_V = |tau - 1[u<0]| u^2,  u = Q_target - V
  Q:  TD regression towards r + gamma (1-d) V(o')
  pi: advantage-weighted BC, weights w = min(exp(beta (Q_target - V)), w_max)

tau -> 1 makes V an optimistic (max-like) estimate without ever querying the policy off-distribution, which is
the whole point: no action is evaluated that the planner did not take.

Decentralised by construction: V and Q consume the agent's own k-masked observation, the same Obs the actor
sees, through their own SetEncoder. The team reward is shared across agents (cooperative, common reward).

NOTE: written without being executed by its author. Run --smoke first.
"""
from __future__ import annotations
import argparse, copy, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from commenv import EnvConfig, CommEnv
from models import SetEncoder, dims, Obs
from bc2 import Rows, add_student_args, student_opts, cfg_from_opts, make_student, save_student, augment2, closed_loop


# ----------------------------------------------------------------------------- data with rewards
class RLRows:
    """Rows plus what offline RL needs: team reward, next observation, terminal flag.

    expert_rows() replays the plan and throws the reward away. This keeps it. Observations are stored one
    time-block after another (S*N rows per step), so the next observation of a row is the SAME row index one
    block later -- no second replay, no extra memory beyond one trailing block."""

    def __init__(self, obs: Obs, a, m, c, r, d, blk):
        self.obs, self.a, self.m, self.c, self.r, self.d, self.blk = obs, a, m, c, r, d, blk

    def __len__(self):
        return self.a.shape[0]

    def to(self, dev):
        return RLRows(Obs(*[x.to(dev) for x in self.obs]), self.a.to(dev), self.m.to(dev), self.c.to(dev),
                      self.r.to(dev), self.d.to(dev), self.blk)

    def index(self, idx):
        return RLRows(self.obs.index(idx), self.a[idx], self.m[idx], self.c[idx], self.r[idx], self.d[idx], self.blk)

    def next_obs(self, idx):
        """Observation after the transition. Terminal rows point at themselves; their target is masked by d."""
        nxt = torch.where(self.d[idx] > 0.5, idx, idx + self.blk)
        return self.obs.index(nxt)


@torch.no_grad()
def rl_rows(seeds, cfg: EnvConfig, actions, channels, H=1) -> RLRows:
    env = CommEnv(cfg, seeds)
    obs = env.reset()
    S, N, T = env.B, cfg.n_agents, cfg.T
    A = torch.as_tensor(actions).float()
    Cc = torch.as_tensor(channels).long()
    Ap = torch.cat([A, torch.zeros(S, max(H - 1, 0), N, 2)], 1)
    O, a_l, m_l, c_l, r_l, d_l = [], [], [], [], [], []
    for t in range(T):
        O.append(obs.flat())
        a_l.append(Ap[:, t:t + H].permute(0, 2, 1, 3).reshape(S * N, H, 2))
        m_l.append(((torch.arange(H) + t) < T).unsqueeze(0).expand(S * N, H))
        c_l.append(Cc[:, t].reshape(-1))
        obs, r, done, _ = env.step(A[:, t], Cc[:, t])
        r_l.append(r.reshape(S, 1).expand(S, N).reshape(-1).float())       # shared team reward
        d_l.append(torch.full((S * N,), 1.0 if t == T - 1 else 0.0))
    O.append(obs.flat())                                                   # trailing block = next_obs of t=T-1
    cat = lambda lst: Obs(*[torch.cat([getattr(o, f) for o in lst], 0) for f in Obs._fields])
    return RLRows(cat(O), torch.cat(a_l), torch.cat(m_l), torch.cat(c_l),
                  torch.cat(r_l), torch.cat(d_l), blk=S * N)


# ----------------------------------------------------------------------------- critics
class VNet(nn.Module):
    def __init__(self, cfg: EnvConfig, d=128):
        super().__init__()
        self.enc = SetEncoder(*dims(cfg), d=d)
        self.head = nn.Linear(d, 1)

    def forward(self, o: Obs):
        return self.head(self.enc(o)).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, cfg: EnvConfig, d=128):
        super().__init__()
        self.C = cfg.n_channels
        self.enc = SetEncoder(*dims(cfg), d=d)
        self.head = nn.Sequential(nn.Linear(d + 2 + cfg.n_channels, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, o: Obs, a, c):
        h = self.enc(o)
        return self.head(torch.cat([h, a, F.one_hot(c, self.C).float()], -1)).squeeze(-1)


def expectile_loss(u, tau):
    return (torch.abs(tau - (u < 0).float()) * u ** 2).mean()


# ----------------------------------------------------------------------------- actor loss, per sample
def awr_loss(model, o: Obs, a_chunk, mask, c, w, w_chunk=0.5, w_ch=1.0):
    """models.StudentActor.bc_loss, but per-sample so the advantage weights can be applied."""
    h = model.enc(o)
    a0 = a_chunk[:, 0]
    if model.mdn_k > 0:
        mu, ls, lw = model._mdn(h)
        z = (a0.unsqueeze(1) - mu) / ls.exp()
        logp = lw - 0.5 * (z ** 2).sum(-1) - ls.sum(-1) - math.log(2 * math.pi)
        pos = -torch.logsumexp(logp, -1)
        mode = mu[torch.arange(h.shape[0], device=h.device), lw.argmax(-1)]
    else:
        mode = model.pos_head(h)
        pos = ((mode - a0) ** 2).mean(-1)
    logits = model.ch_head(h)
    per = pos + w_ch * F.cross_entropy(logits, c, reduction="none")
    if model.chunk > 1:
        pred = model.chunk_head(h).view(-1, model.chunk - 1, 2)
        m = mask[:, 1:model.chunk].float().unsqueeze(-1)
        per = per + w_chunk * (((pred - a_chunk[:, 1:model.chunk]) ** 2) * m).sum((1, 2)) / (2 * m.sum(dim=(1, 2)) + 1e-6)
    stats = dict(mse0=float(F.mse_loss(mode, a0).detach()), acc=float((logits.argmax(-1) == c).float().mean().detach()))
    return (w * per).mean(), stats


# ----------------------------------------------------------------------------- training
def train_iql(model, ema, rows: RLRows, cfg, dev, a, log=print):
    n = len(rows)
    rows = rows.to(dev)
    V, Q = VNet(cfg, a.d).to(dev), QNet(cfg, a.d).to(dev)
    Qt = copy.deepcopy(Q).to(dev)
    for p in Qt.parameters():
        p.requires_grad_(False)
    optV = torch.optim.AdamW(V.parameters(), lr=a.critic_lr, weight_decay=1e-4)
    optQ = torch.optim.AdamW(Q.parameters(), lr=a.critic_lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(a.seed)
    idx_all = torch.arange(n, device=dev)

    # ---- phase 1: critics
    t0 = time.time()
    for ep in range(a.critic_epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            r = rows.index(idx)
            nobs = rows.next_obs(idx)
            with torch.no_grad():
                q_t = Qt(r.obs, r.a[:, 0], r.c)
            lv = expectile_loss(q_t - V(r.obs), a.tau)
            optV.zero_grad(set_to_none=True); lv.backward()
            nn.utils.clip_grad_norm_(V.parameters(), 1.0); optV.step()
            with torch.no_grad():
                target = r.r + a.gamma * (1.0 - r.d) * V(nobs)
            lq = F.mse_loss(Q(r.obs, r.a[:, 0], r.c), target)
            optQ.zero_grad(set_to_none=True); lq.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), 1.0); optQ.step()
            with torch.no_grad():
                for pt, p in zip(Qt.parameters(), Q.parameters()):
                    pt.mul_(1 - a.polyak).add_(p.detach(), alpha=a.polyak)
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"    critic ep {ep+1:3d}/{a.critic_epochs}  L_V {float(lv):8.4f}  L_Q {float(lq):8.4f}  {time.time()-t0:.0f}s")

    # ---- advantage weights, computed once with the frozen critics
    with torch.no_grad():
        A_ = []
        for i in range(0, n, 8192):
            idx = idx_all[i:i + 8192]
            r = rows.index(idx)
            A_.append(Qt(r.obs, r.a[:, 0], r.c) - V(r.obs))
        adv = torch.cat(A_)
        log(f"    advantages: mean {float(adv.mean()):+.3f}  sd {float(adv.std()):.3f}  "
            f"p1 {float(adv.quantile(0.01)):+.3f}  p99 {float(adv.quantile(0.99)):+.3f}")
        if a.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        W = torch.exp(a.beta * adv).clamp(max=a.w_max)
        W = W / W.mean()
        n_eff = float(W.sum() ** 2 / (W ** 2).sum())
        log(f"    AWR weights: mean {float(W.mean()):.3f}  p50 {float(W.median()):.3f}  "
            f"p99 {float(W.quantile(0.99)):.3f}  max {float(W.max()):.3f}  "
            f"n_eff {n_eff:.0f}/{n} ({100 * n_eff / n:.1f}%)")

    # ---- phase 2: advantage-weighted behaviour cloning
    model.to(dev).train(); ema.to(dev)
    spe = math.ceil(n / a.bs); total = max(1, a.epochs0 * spe); warm = max(1, int(0.03 * total))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    step, t0 = 0, time.time()
    for ep in range(a.epochs0):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            r = rows.index(idx)
            ob, aa = (augment2(r.obs, r.a, g, a.obs_extra) if not a.no_aug else (r.obs, r.a))
            loss, st = awr_loss(model, ob, aa, r.m, r.c, W[idx], w_chunk=a.w_chunk)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            dcy = min(0.999, (1 + step) / (10 + step)); step += 1
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(dcy).add_(p.detach(), alpha=1 - dcy)
        if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == a.epochs0:
            log(f"    actor  ep {ep+1:4d}/{a.epochs0}  loss {float(loss.detach()):8.4f}  mse0 {st['mse0']:.4f} "
                f"acc {st['acc']:.3f}  {time.time()-t0:.0f}s")
    return ema


def main():
    ap = argparse.ArgumentParser()
    add_student_args(ap)
    ap.add_argument("--name", default="iql")
    ap.add_argument("--out", default="results2")
    ap.add_argument("--tau", type=float, default=0.8, help="expectile; ->1 is more optimistic")
    ap.add_argument("--beta", type=float, default=3.0, help="AWR inverse temperature")
    ap.add_argument("--w_max", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--polyak", type=float, default=0.005)
    ap.add_argument("--critic_epochs", type=int, default=60)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--n_monitor", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--adv_norm", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs0, a.critic_epochs, a.n_monitor = 2, 2, 8

    opts = student_opts(a)
    cfg = cfg_from_opts(opts)
    run_dir = os.path.join(a.out, a.name, f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(*s):
        print(*s, flush=True); print(*s, file=logf, flush=True)

    log(f"== {a.name} seed {a.seed}  opts={opts}  device={a.device}  "
        f"iql(tau={a.tau}, beta={a.beta}, gamma={a.gamma}, critic_epochs={a.critic_epochs})")
    d = torch.load(a.train_data, map_location="cpu", weights_only=False)
    seeds = d["seeds"][:60] if a.smoke else d["seeds"]
    rows = rl_rows(seeds, cfg, d["actions"][:len(seeds)], d["channels"][:len(seeds)], H=opts["chunk"])
    log(f"rows: {len(rows)}  (demonstrator return {float(torch.as_tensor(d['expert_return']).mean()):.2f})")

    torch.manual_seed(a.seed)
    model = make_student(opts)
    ema = copy.deepcopy(model)
    ema = train_iql(model, ema, rows, cfg, a.device, a, log=log)

    mon = list(range(700_000, 700_000 + a.n_monitor))
    ret = float(closed_loop(ema.cpu().eval(), mon, cfg).mean())
    log(f"[iql] closed-loop {ret:.2f} on {a.n_monitor} monitor scenarios")
    save_student(os.path.join(run_dir, "best.pt"), ema, opts)
    json.dump(dict(name=a.name, seed=a.seed, opts=opts, monitor_return_best=ret,
                   iql=dict(tau=a.tau, beta=a.beta, gamma=a.gamma, critic_epochs=a.critic_epochs)),
              open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"saved {run_dir}/best.pt")


if __name__ == "__main__":
    main()