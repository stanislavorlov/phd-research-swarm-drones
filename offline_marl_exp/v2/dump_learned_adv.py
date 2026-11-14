"""dump_learned_adv.py -- per-row learned IQL advantages on the paired dataset, saved next to the exact ones, for
Figure 2. iql.py computes these advantages but never writes them out, so this re-trains the critic exactly as
iql.py's learned_weights() does (same classes, losses, batch order and RNG sequence) and saves the result.
No actor is trained.

    python3 dump_learned_adv.py --seed 0 --device cpu        # ~ the critic half of one L125_pairs run
    python3 dump_learned_adv.py --seed 0 --critic_epochs 2   # smoke test, a minute or two

Writes data/learned_adv_s<seed>.npz with
    learned_adv [96000]   Q_target(o, a, c) - V(o), pair_rows order: (t, branch, scenario, agent)
    src         [96000]   0 = branch A (committed, strong), 1 = branch B (evaluated, weak)
    exact_adv_a/_b [S, T] from data/adv_train.pt (solver_advantages.py), per state
    S, T, N
Check against the run's log: the printed branch gap should be ~0.000 and the learned-advantage sd ~0.7, as in
results2/L125_pairs/seed<s>/log.txt ("adv by branch ... gap"). A different device gives a different critic, so
the level will not match to the digit; the gap is what the figure is about.
"""
from __future__ import annotations
import argparse, copy, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iql import pair_rows, VNet, QNet, fm, expectile_loss
from bc2 import add_student_args, student_opts, cfg_from_opts, make_student


def main():
    ap = argparse.ArgumentParser()
    add_student_args(ap)                                   # --seed, --device, --d, --bs, --k, ... as iql.py
    ap.add_argument("--pairs_data", default="data/pairs_train.pt")
    ap.add_argument("--solver_adv", default="data/adv_train.pt")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--polyak", type=float, default=0.005)
    ap.add_argument("--critic_epochs", type=int, default=60)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--critic_full", action="store_true")
    ap.add_argument("--out", default=None)
    # iql.py's L*_pairs runs used these student settings; set them here so the command line stays short
    ap.set_defaults(k=6.0, k_a=-1.0, bcast=6, head="mdn", d=128, mdn_k=5, chunk=4)
    a = ap.parse_args()
    opts = student_opts(a)
    cfg = cfg_from_opts(opts)
    dev = a.device

    d = torch.load(a.pairs_data, map_location="cpu", weights_only=False)
    rows = pair_rows(d["seeds"], cfg, d["actions_a"], d["channels_a"], d["actions_b"], d["channels_b"], H=opts["chunk"])
    n, S, N, T = len(rows), len(d["seeds"]), cfg.n_agents, cfg.T
    assert n == T * 2 * S * N, (n, T, S, N)
    print(f"rows {n}  (S={S}, T={T}, N={N})  device={dev}")

    torch.manual_seed(a.seed)
    make_student(opts)                                     # iql.main builds the actor first; keep the RNG in step
    rows = rows.to(dev)
    V, Q = VNet(cfg, a.d).to(dev), QNet(cfg, a.d).to(dev)
    Qt = copy.deepcopy(Q).to(dev)
    for p in Qt.parameters():
        p.requires_grad_(False)
    optV = torch.optim.AdamW(V.parameters(), lr=a.critic_lr, weight_decay=1e-4)
    optQ = torch.optim.AdamW(Q.parameters(), lr=a.critic_lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(a.seed)
    idx_all = torch.arange(n, device=dev)
    t0 = time.time()
    for ep in range(a.critic_epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            r = rows.index(idx)
            nobs = rows.next_obs(idx)
            um, am = fm(r.obs, idx, N) if a.critic_full else (None, None)
            with torch.no_grad():
                q_t = Qt(r.obs, r.a[:, 0], r.c, um, am)
            lv = expectile_loss(q_t - V(r.obs, um, am), a.tau)
            optV.zero_grad(set_to_none=True); lv.backward()
            nn.utils.clip_grad_norm_(V.parameters(), 1.0); optV.step()
            with torch.no_grad():
                target = r.r + a.gamma * (1.0 - r.d) * V(nobs, um, am)
            lq = F.mse_loss(Q(r.obs, r.a[:, 0], r.c, um, am), target)
            optQ.zero_grad(set_to_none=True); lq.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), 1.0); optQ.step()
            with torch.no_grad():
                for pt, p in zip(Qt.parameters(), Q.parameters()):
                    pt.mul_(1 - a.polyak).add_(p.detach(), alpha=a.polyak)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  critic ep {ep+1:3d}/{a.critic_epochs}  L_V {float(lv):8.4f}  L_Q {float(lq):8.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        A_ = []
        for i in range(0, n, 8192):
            idx = idx_all[i:i + 8192]
            r = rows.index(idx)
            um, am = fm(r.obs, idx, N) if a.critic_full else (None, None)
            A_.append(Qt(r.obs, r.a[:, 0], r.c, um, am) - V(r.obs, um, am))
        adv = torch.cat(A_).cpu().numpy()
    src = rows.src.cpu().numpy()
    L = adv.reshape(T, 2, S, N).mean(-1)
    gap = (L[:, 0] - L[:, 1]).ravel()
    print(f"learned advantages: mean {adv.mean():+.3f}  sd {adv.std():.3f}   "
          f"branch gap {adv[src < 0.5].mean() - adv[src > 0.5].mean():+.4f} (per-state sd {gap.std():.4f})")

    e = torch.load(a.solver_adv, map_location="cpu", weights_only=False)
    ea, eb = np.asarray(e["adv_a"], np.float32), np.asarray(e["adv_b"], np.float32)
    assert ea.shape == (S, T), (ea.shape, S, T)
    print(f"exact advantages:   branch gap {(ea - eb).mean():+.3f}  (sd {(ea - eb).std():.3f}, "
          f"weak better at {100 * ((ea - eb) < 0).mean():.1f}%)")
    out = a.out or f"data/learned_adv_s{a.seed}.npz"
    np.savez(out, learned_adv=adv.astype(np.float32), src=src.astype(np.int8), exact_adv_a=ea, exact_adv_b=eb,
             S=S, T=T, N=N, seed=a.seed, critic_epochs=a.critic_epochs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
