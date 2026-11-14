"""supervised_q.py -- is the learned critic's zero branch gap a representation failure or an objective failure?

Regress the critic architecture of iql.py DIRECTLY onto the planner's exact values on the paired dataset. No TD,
no expectile, no bootstrapping: plain supervised MSE with the targets of solver_advantages.py. Three arms, one
command each, same rows, same encoder, same head, same optimiser and budget as the IQL critic (60 epochs, AdamW 3e-4):

  --target q            y = Q_true(s, a_b) = r + J(s')      the value IQL is trying to learn, minus its objective
  --target adv          y = A(s, a_b) = +-(Q_A - Q_B)/2     the same network, with the state part removed from the
                                                            target: the only thing left to fit is the action effect
  --target adv --branch_bit   adds a 1-bit branch indicator to the head input: a ceiling, the gap the architecture
                              can carry when the action input is perfectly informative

Reading (pre-registered in the paper draft, §4.5):
  q ~ 0, adv ~ 0, bit > 0  -> the agent's OWN action does not carry the team gap (per-agent credit), not architecture
  q ~ 0, adv > 0           -> representation is adequate; MSE on a state-dominated target drowns the action term
                              (an objective/conditioning failure that TD + expectile inherit, not IQL-specific)
  q > 0                    -> supervision alone recovers it; the culprit is the TD/expectile objective
Held-out scenarios (--n_val) are excluded from training and reported separately.

    python3 supervised_q.py --target q   --seed 0 --device cpu --epochs 2        # smoke
    python3 supervised_q.py --target q   --seed 0 --device cpu
    python3 supervised_q.py --target adv --seed 0 --device cpu
    python3 supervised_q.py --target adv --branch_bit --seed 0 --device cpu

Writes results2/SQ_<target>[_bit]/seed<s>/{log.txt, summary.json}.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iql import pair_rows, fm
from models import SetEncoder, dims, Obs
from bc2 import add_student_args, student_opts, cfg_from_opts, make_student


class QHead(nn.Module):
    """iql.QNet, verbatim, with an optional extra input column. With extra = 0 it is the IQL critic."""

    def __init__(self, cfg, d=128, extra=0):
        super().__init__()
        self.C = cfg.n_channels
        self.enc = SetEncoder(*dims(cfg), d=d)
        self.head = nn.Sequential(nn.Linear(d + 2 + cfg.n_channels + extra, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, o: Obs, a, c, um=None, am=None, bit=None):
        h = self.enc(o, um, am)
        z = [h, a, F.one_hot(c, self.C).float()] + ([bit.unsqueeze(-1)] if bit is not None else [])
        return self.head(torch.cat(z, -1)).squeeze(-1)


def per_state(pred, T, S, N):
    """rows are (t, branch, scenario, agent); return per-state branch means [S, T] for A and B."""
    P = pred.reshape(T, 2, S, N).mean(-1)
    return P[:, 0].T, P[:, 1].T


def gap_metrics(pA, pB, qa, qb, sel):
    lg, eg = (pA - pB)[sel].ravel(), (qa - qb)[sel].ravel()
    slope = float(np.polyfit(eg, lg, 1)[0])
    return dict(learned_gap=float(lg.mean()), exact_gap=float(eg.mean()), learned_gap_sd=float(lg.std()),
                corr=float(np.corrcoef(eg, lg)[0, 1]), slope=slope,
                sign_acc=float(((lg > 0) == (eg > 0)).mean()), always_A_acc=float((eg > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    add_student_args(ap)
    ap.add_argument("--pairs_data", default="data/pairs_train.pt")
    ap.add_argument("--solver_adv", default="data/adv_train.pt")
    ap.add_argument("--target", choices=["q", "adv"], required=True)
    ap.add_argument("--branch_bit", action="store_true")
    ap.add_argument("--critic_full", action="store_true", help="full-state masks, as iql.py --critic_full")
    ap.add_argument("--epochs", type=int, default=60, help="the IQL critic's budget")
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--n_val", type=int, default=60, help="held-out scenarios (the last n_val of the 300)")
    ap.add_argument("--out", default="results2")
    ap.set_defaults(k=6.0, k_a=-1.0, bcast=6, head="mdn", d=128, mdn_k=5, chunk=4)
    a = ap.parse_args()
    opts = student_opts(a)
    cfg = cfg_from_opts(opts)
    dev = a.device
    name = f"SQ_{a.target}" + ("_bit" if a.branch_bit else "") + ("_full" if a.critic_full else "")
    run_dir = os.path.join(a.out, name, f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(*s):
        print(*s, flush=True); print(*s, file=logf, flush=True)

    log(f"== {name} seed {a.seed}  opts={opts}  device={dev}  epochs={a.epochs}  lr={a.critic_lr}  n_val={a.n_val}")

    d = torch.load(a.pairs_data, map_location="cpu", weights_only=False)
    e = torch.load(a.solver_adv, map_location="cpu", weights_only=False)
    rows = pair_rows(d["seeds"], cfg, d["actions_a"], d["channels_a"], d["actions_b"], d["channels_b"], H=1)
    n, S, N, T = len(rows), len(d["seeds"]), cfg.n_agents, cfg.T
    assert n == T * 2 * S * N, (n, T, S, N)
    qa, qb = np.asarray(e["q_a"], np.float64), np.asarray(e["q_b"], np.float64)        # [S, T]
    assert qa.shape == (S, T), (qa.shape, S, T)
    gap = qa - qb

    # target per row, in pair_rows order (t, branch, scenario, agent)
    if a.target == "q":
        Y = np.stack([qa.T, qb.T], 1)                                                    # [T, 2, S]
    else:
        Y = np.stack([gap.T / 2, -gap.T / 2], 1)
    Y = torch.as_tensor(np.repeat(Y[..., None], N, -1).reshape(-1), dtype=torch.float32)
    scen = torch.arange(S).view(1, 1, S, 1).expand(T, 2, S, N).reshape(-1)
    train = scen < S - a.n_val
    sel_tr, sel_va = np.arange(S) < S - a.n_val, np.arange(S) >= S - a.n_val

    # how much of the target's variance is the action at all?  (no training involved)
    Qall = np.stack([qa, qb], -1)
    within = float(((gap / 2) ** 2).mean())                       # variance of the branch deviation about the pair mean
    total = float(Qall.var())
    log(f"rows {n} (S={S}, T={T}, N={N}); train scenarios {S - a.n_val}, held-out {a.n_val}")
    log(f"exact Q: var {total:.2f};  within-pair (action) part {within:.3f} = {100 * within / total:.2f}% of it;  "
        f"gap {gap.mean():+.3f} (sd {gap.std():.3f}), weak better at {100 * (gap < 0).mean():.1f}%")

    mu, sd = float(Y[train].mean()), float(Y[train].std())
    Yn = ((Y - mu) / sd).to(dev)
    bit = (1.0 - rows.src).to(dev) if a.branch_bit else None           # 1 on branch A rows
    rows = rows.to(dev)

    torch.manual_seed(a.seed)
    make_student(opts)                                                  # keep the RNG in step with iql.py
    Q = QHead(cfg, a.d, extra=int(a.branch_bit)).to(dev)
    opt = torch.optim.AdamW(Q.parameters(), lr=a.critic_lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(a.seed)
    tr_idx = torch.nonzero(train).squeeze(1)
    t0 = time.time()

    def predict():
        Q.eval(); out = []
        with torch.no_grad():
            for i in range(0, n, 8192):
                idx = torch.arange(i, min(n, i + 8192), device=dev)
                r = rows.index(idx)
                um, am = fm(r.obs, idx, N) if a.critic_full else (None, None)
                out.append(Q(r.obs, r.a[:, 0], r.c, um, am, None if bit is None else bit[idx]))
        Q.train()
        return (torch.cat(out).cpu().numpy() * sd + mu)

    def report(tag):
        P = predict()
        pA, pB = per_state(P, T, S, N)
        m_tr, m_va = gap_metrics(pA, pB, qa, qb, sel_tr), gap_metrics(pA, pB, qa, qb, sel_va)
        ytrue = Y.numpy()
        va_rows = ~train.numpy()
        r2 = 1 - float(((P - ytrue)[va_rows] ** 2).mean()) / float(ytrue[va_rows].var())
        log(f"  [{tag}] held-out R2 {r2:+.3f} | learned gap train {m_tr['learned_gap']:+.3f}  "
            f"held-out {m_va['learned_gap']:+.3f}  (exact {m_va['exact_gap']:+.3f})  corr {m_va['corr']:+.3f}  "
            f"slope {m_va['slope']:+.3f}  sign acc {m_va['sign_acc']:.3f} (always-A {m_va['always_A_acc']:.3f})")
        return dict(r2_heldout=r2, train=m_tr, heldout=m_va)

    for ep in range(a.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)].to(dev)
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            r = rows.index(idx)
            um, am = fm(r.obs, idx, N) if a.critic_full else (None, None)
            loss = F.mse_loss(Q(r.obs, r.a[:, 0], r.c, um, am, None if bit is None else bit[idx]), Yn[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), 1.0); opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"  ep {ep+1:3d}/{a.epochs}  mse(norm) {float(loss):.4f}  {time.time()-t0:.0f}s")
        if (ep + 1) % 20 == 0 and ep + 1 < a.epochs:
            report(f"ep {ep+1}")
    res = report("final")
    json.dump(dict(name=name, seed=a.seed, target=a.target, branch_bit=a.branch_bit, critic_full=a.critic_full,
                   epochs=a.epochs, action_variance_share=within / total, **res),
              open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"saved {run_dir}/summary.json")


if __name__ == "__main__":
    main()
