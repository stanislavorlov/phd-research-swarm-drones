"""
bc2.py -- better student training: bigger budget, EMA, action chunks / mixture head, DAgger data collection.

Everything here is additive: bc.py, models.Actor and mappo.py are unchanged, and a StudentActor checkpoint plugs
straight into the MAPPO phase (its forward() is a Gaussian-around-the-mode, like Actor).

Rows are ordered like env.obs().flat():   row = (t, scenario b, agent n)  ->  b*N + n within a step.
"""
from __future__ import annotations
import copy
import math
import time
from dataclasses import dataclass
import numpy as np
import torch

from commenv import EnvConfig, CommEnv, Obs
from models import StudentActor, NetPolicy
from evalproto import returns
from bc import _D8
from planner_state import plan_from_state


# ----------------------------------------------------------------------------- data
@dataclass
class Rows:
    obs: Obs
    a: torch.Tensor      # [n,H,2]  expert displacements for steps t..t+H-1 (unit ball)
    m: torch.Tensor      # [n,H]    bool, step t+j exists in the (remaining) plan
    c: torch.Tensor      # [n]      expert channel at step t (canonicalised w.r.t. the current channel)

    def __len__(self):
        return self.a.shape[0]

    def index(self, idx):
        return Rows(self.obs.index(idx), self.a[idx], self.m[idx], self.c[idx])

    def to(self, dev):
        return Rows(Obs(*[x.to(dev) for x in self.obs]), self.a.to(dev), self.m.to(dev), self.c.to(dev))

    @staticmethod
    def cat(lst):
        return Rows(Obs(*[torch.cat([getattr(r.obs, f) for r in lst], 0) for f in Obs._fields]),
                    torch.cat([r.a for r in lst]), torch.cat([r.m for r in lst]), torch.cat([r.c for r in lst]))


def expert_rows(seeds, cfg: EnvConfig, actions, channels, H=1) -> Rows:
    """Open-loop expert dataset (expert.py output) -> rows with the env's own observations and H-step chunks."""
    env = CommEnv(cfg, seeds); obs = env.reset()
    S, N, T = env.B, cfg.n_agents, cfg.T
    A = torch.as_tensor(actions).float(); Cc = torch.as_tensor(channels).long()      # [S,T,N,2], [S,T,N]
    Ap = torch.cat([A, torch.zeros(S, max(H - 1, 0), N, 2)], 1)
    O, a_l, m_l, c_l = [], [], [], []
    for t in range(T):
        O.append(obs.flat())
        a_l.append(Ap[:, t:t + H].permute(0, 2, 1, 3).reshape(S * N, H, 2))
        m_l.append(((torch.arange(H) + t) < T).unsqueeze(0).expand(S * N, H))
        c_l.append(Cc[:, t].reshape(-1))
        obs, _, _, _ = env.step(A[:, t], Cc[:, t])
    return Rows(Obs(*[torch.cat([getattr(o, f) for o in O], 0) for f in Obs._fields]),
                torch.cat(a_l), torch.cat(m_l), torch.cat(c_l))


@torch.no_grad()
def dagger_collect(actor_cpu, seeds, cfg: EnvConfig, cfg_plan: EnvConfig, beta, plan_kw, H=1, noise=0.05, rng_seed=0):
    """
    Roll the (EMA) student out; at EVERY visited state ask the receding-horizon expert (planner restarted from that
    state with the remaining horizon) and record its plan head as the label. With prob. `beta` (per scenario, per step)
    the EXPERT's action is executed instead of the student's (DAgger mixture); `noise` is exploration noise on the
    student's executed displacement. Returns (Rows, mean planning seconds per step, mean rollout return).
    """
    g = torch.Generator().manual_seed(rng_seed)
    env = CommEnv(cfg, seeds); obs = env.reset()
    S, N, T = env.B, cfg.n_agents, cfg.T
    rows, solve, ret = [], 0.0, torch.zeros(S)
    for t0 in range(T):
        out = plan_from_state(env.users_traj[:, t0 + 1:], env.uav, env.ch, cfg_plan, **plan_kw)
        solve += out["solve_s"]
        pa, pc = out["actions"], out["channels"]                                     # [S,h,N,2], [S,h,N]
        h = pa.shape[1]
        chunk = torch.zeros(S, H, N, 2)
        k = min(H, h)
        chunk[:, :k] = pa[:, :k]
        rows.append(Rows(obs.flat(), chunk.permute(0, 2, 1, 3).reshape(S * N, H, 2),
                         (torch.arange(H) < h).unsqueeze(0).expand(S * N, H).clone(), pc[:, 0].reshape(-1)))
        mu, logits = actor_cpu(obs.flat())
        a_s = mu.view(S, N, 2) + noise * torch.randn(S, N, 2, generator=g)
        c_s = logits.argmax(-1).view(S, N)
        use_exp = (torch.rand(S, generator=g) < beta)
        a_x = torch.where(use_exp.view(S, 1, 1), pa[:, 0], a_s)
        c_x = torch.where(use_exp.view(S, 1), pc[:, 0], c_s)
        obs, r, _, _ = env.step(a_x, c_x)
        ret += r
    return Rows.cat(rows), solve / T, float(ret.mean())


# ----------------------------------------------------------------------------- training
def augment2(o: Obs, a, g: torch.Generator, extra: bool):
    """Random element of the square's symmetry group (8x data multiplier); handles the obs_extra features and
    action chunks a [n,H,2]. Physics (square world, isotropic path loss, reflecting boundaries) is invariant."""
    n, dev = a.shape[0], a.device
    M = _D8.to(dev)[torch.randint(0, 8, (n,), generator=g).to(dev)]                  # [n,2,2]
    mv = lambda x: torch.einsum("nij,n...j->n...i", M, x)
    sf = o.sf.clone(); sf[:, :2] = mv(sf[:, :2] - 0.5) + 0.5
    if extra:
        sf[:, -2:] = mv(sf[:, -2:])
    parts = [mv(o.uf[..., 0:2]), mv(o.uf[..., 2:4])]
    if extra:
        parts.append(mv(o.uf[..., 4:6]))
    af = torch.cat([mv(o.af[..., 0:2]), o.af[..., 2:]], -1)
    return Obs(sf, torch.cat(parts, -1), o.um, af, o.am), mv(a)


@torch.no_grad()
def eval_rows(model, rows: Rows, dev, bs=8192):
    model.eval()
    tot = dict(mse0=0.0, acc=0.0); n = len(rows)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=dev)
        r = rows.index(idx)
        _, st = model.bc_loss(r.obs, r.a, r.m, r.c)
        for k in tot:
            tot[k] += st[k] * len(idx)
    return {k: v / n for k, v in tot.items()}


def train_student(model, ema, rows: Rows, val: Rows, epochs, dev, bs=2048, lr=2e-3, wd=1e-4, aug=True, extra=False,
                  seed=0, ema_decay=0.999, w_chunk=0.5, log_every=10, log=print):
    """Trains `model` in place and keeps `ema` (the model to evaluate / roll out / save). Cosine LR with warm-up."""
    model.to(dev).train(); ema.to(dev)
    rows_d, val_d = rows.to(dev), val.to(dev)
    n = len(rows_d)
    spe = math.ceil(n / bs); total = max(1, epochs * spe); warm = max(1, int(0.03 * total))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    g = torch.Generator().manual_seed(seed)
    sub = rows_d.index(torch.randperm(n, generator=g)[:min(20000, n)].to(dev))
    step, t0 = 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            r = rows_d.index(perm[i:i + bs])
            ob, a = (augment2(r.obs, r.a, g, extra) if aug else (r.obs, r.a))
            loss, _ = model.bc_loss(ob, a, r.m, r.c, w_chunk=w_chunk)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            dcy = min(ema_decay, (1 + step) / (10 + step)); step += 1
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(dcy).add_(p.detach(), alpha=1 - dcy)
        if (ep + 1) % log_every == 0 or ep == 0 or ep + 1 == epochs:
            tr, va = eval_rows(ema, sub, dev), eval_rows(ema, val_d, dev)
            log(f"    ep {ep+1:4d}/{epochs}  loss {float(loss):8.4f} | EMA train mse0 {tr['mse0']:.4f} acc {tr['acc']:.3f} | "
                f"val mse0 {va['mse0']:.4f} acc {va['acc']:.3f} | {time.time()-t0:.0f}s")
    return ema


# ----------------------------------------------------------------------------- eval helpers
def closed_loop(model, seeds, cfg: EnvConfig):
    m = copy.deepcopy(model).cpu().eval()
    return returns(NetPolicy(m), seeds, cfg)[0]


def expert_returns_on(seeds, cfg: EnvConfig, cfg_plan: EnvConfig, plan_kw, chunk=100):
    """Strong planner from the initial state over the whole episode, replayed through the ENV (cfg) for the return."""
    out = []
    for i in range(0, len(seeds), chunk):
        s = seeds[i:i + chunk]
        env = CommEnv(cfg, s); env.reset()
        p = plan_from_state(env.users_traj[:, 1:], env.uav0, env.ch0, cfg_plan, **plan_kw)
        ret = torch.zeros(env.B)
        for t in range(cfg.T):
            _, r, _, _ = env.step(p["actions"][:, t], p["channels"][:, t])
            ret += r
        out.append(ret.numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- student construction / io
def add_student_args(ap):
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--head", choices=["mse", "mdn"], default="mse")
    ap.add_argument("--mdn_k", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=1, help="1 = off; H>1 adds an auxiliary head for the next H-1 plan steps")
    ap.add_argument("--obs_extra", action="store_true", help="previous displacement + extrapolated user positions")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--epochs0", type=int, default=150)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--w_chunk", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_data", default="data/expert_train.pt")
    ap.add_argument("--val_data", default="data/expert_val.pt")
    ap.add_argument("--plan_restarts", type=int, default=4)
    ap.add_argument("--plan_rounds", type=int, default=4)
    ap.add_argument("--plan_iters", type=int, default=80)
    ap.add_argument("--plan_energy", type=float, default=None, help="energy price used by the PLANNER (env keeps 0.5)")


def student_opts(a):
    return dict(k=a.k, d=a.d, mdn_k=(a.mdn_k if a.head == "mdn" else 0), chunk=a.chunk, obs_extra=bool(a.obs_extra))


def make_student(opts) -> StudentActor:
    cfg = EnvConfig(k=opts["k"], obs_extra=opts["obs_extra"])
    return StudentActor(cfg, d=opts["d"], mdn_k=opts["mdn_k"], chunk=opts["chunk"])


def save_student(path, model, opts):
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, opts=opts), path)


def load_student(path):
    ck = torch.load(path, weights_only=False)
    m = make_student(ck["opts"]); m.load_state_dict(ck["state"])
    return m.eval(), EnvConfig(k=ck["opts"]["k"], obs_extra=ck["opts"]["obs_extra"])
