"""
run_madt.py -- train and score a Multi-Agent Decision Transformer, with the same conventions as student.py.

    # the main arm: return-conditioned on the mixed-quality dataset
    python3 run_madt.py --name M_mix --seed 0 \
        --train_data data/expert_train_mix.pt --val_data data/expert_val_mix.pt \
        --k 6 --k_a -1 --bcast 6 --d 128 --epochs 25 --targets 100 120 140 160

    # score the selected arm on the frozen test set
    python3 run_madt.py --name M_mix --seed 0 --eval_only --seed0 900000 --n_final 1024 --planner strong++

    python3 run_madt.py --smoke --name SMOKE --train_data ... --val_data ...

Why a separate runner: `final_eval.py` loads checkpoints with `bc2.load_student`, which builds a StudentActor.
MADT is a different class with a different rollout protocol (it carries per-agent history and a return-to-go),
so it cannot go through that path; this script does its own closed-loop evaluation instead. It writes
`results2/<name>/seed<s>/summary.json` with `monitor_return_best`, so `collect_p2.py` picks the arm up unchanged.

`train()` is `madt.train_madt` with the model capacity, epoch count and batch size exposed, so that a weak result
cannot be dismissed as undertrained. Passing --d 64 --layers 3 --heads 4 --epochs 25 --madt_bs 48 reproduces
`madt.train_madt` exactly; the loss and the schedule are unchanged.

**The return-to-go target is a hyperparameter**, and a decision transformer's closed-loop score depends on it
strongly. --targets sweeps it and selects on the MONITOR seeds only (700000+), which is the project's model
selection rule; dev and test are never used to choose it.

NOTE: written without being executed by its author. Run --smoke first.
"""
from __future__ import annotations
import argparse, copy, glob, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from commenv import EnvConfig
from evalproto import returns, seed_level_ci
from bc2 import add_student_args, student_opts, cfg_from_opts
from madt import MADT, MADTPolicy, build_sequences, action_tokens


def load_traj(path, n=None):
    d = torch.load(path, weights_only=False)
    s, a, c = d["seeds"], d["actions"], d["channels"]
    if n:
        s, a, c = s[:n], a[:n], c[:n]
    return s, a, c, d.get("cfg")


def train(cfg, train_path, val_path, seed=0, epochs=25, bs=48, lr=1e-3, d=64, layers=3, heads=4,
          rtg_scale=100.0, n_cap=None, log=print):
    """madt.train_madt with capacity and schedule exposed. Loss and LR schedule are unchanged."""
    torch.manual_seed(seed)
    tr = load_traj(train_path, n_cap)
    va = load_traj(val_path, n_cap)
    for name, got in (("train", tr[3]), ("val", va[3])):
        if got is not None:
            import bc2
            bc2.assert_env_match(cfg, got, f"{name} dataset")
    o_tr, a_tr, c_tr, r_tr = build_sequences(tr[0], cfg, tr[1], tr[2])
    o_va, a_va, c_va, r_va = build_sequences(va[0], cfg, va[1], va[2])
    model = MADT(cfg, d=d, layers=layers, heads=heads, rtg_scale=rtg_scale)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = a_tr.shape[0]
    steps = max(1, epochs * ((n + bs - 1) // bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    C = cfg.n_channels
    t0 = time.time()
    log(f"    sequences: {n} (= scenarios x {cfg.n_agents} agents), rtg range "
        f"[{float(r_tr.min()):.1f}, {float(r_tr.max()):.1f}]")

    def loss_fn(o, a, c, r):
        pos, logits = model(o, action_tokens(a, c, C), r)
        mse = F.mse_loss(pos, a)
        return mse + F.cross_entropy(logits.reshape(-1, C), c.reshape(-1)), float(mse)

    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, _ = loss_fn(o_tr.index(idx), a_tr[idx], c_tr[idx], r_tr[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == epochs:
            with torch.no_grad():
                M = o_va.sf.shape[0]
                vl = float(np.mean([loss_fn(o_va.index(torch.arange(i, min(i + 256, M))),
                                            a_va[i:i + 256], c_va[i:i + 256], r_va[i:i + 256])[1]
                                    for i in range(0, M, 256)]))
            log(f"    ep {ep+1:3d}/{epochs}  val pos-MSE {vl:.4f}  ({time.time()-t0:.0f}s)")
    return model


def score(model, target, seeds, cfg):
    r, o = returns(MADTPolicy(model.eval(), target), seeds, cfg)
    return float(r.mean()), float(o.mean())


def main():
    ap = argparse.ArgumentParser()
    add_student_args(ap)                       # k, k_a, bcast, obs_extra, d, seed, device, env size, data paths
    ap.add_argument("--name", default="madt")
    ap.add_argument("--out", default="results2")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--rtg_scale", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--madt_bs", type=int, default=48, help="separate from --bs, which student.py sizes for BC")
    ap.add_argument("--madt_lr", type=float, default=1e-3)
    ap.add_argument("--targets", type=float, nargs="*", default=[100, 120, 140],
                    help="return-to-go targets to sweep; selected on the MONITOR seeds only")
    ap.add_argument("--n_monitor", type=int, default=128)
    ap.add_argument("--eval_only", action="store_true", help="load best.pt and score it; no training")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed0", type=int, default=800_000)
    ap.add_argument("--n_final", type=int, default=256)
    ap.add_argument("--planner", default="strong++")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.n_monitor, a.targets = 2, 8, [120]

    opts = student_opts(a)
    cfg = cfg_from_opts(opts)
    run_dir = os.path.join(a.out, a.name, f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(*s):
        print(*s, flush=True); print(*s, file=logf, flush=True)

    log(f"== {a.name} seed {a.seed}  MADT d={a.d} layers={a.layers} heads={a.heads} "
        f"epochs={a.epochs} bs={a.madt_bs} lr={a.madt_lr}  opts={opts}")
    mon = list(range(700_000, 700_000 + a.n_monitor))
    ckpt = os.path.join(run_dir, "best.pt")

    if a.eval_only:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = MADT(cfg, d=sd["opts"].get("d", 64), layers=sd["layers"], heads=sd["heads"],
                     rtg_scale=sd["rtg_scale"])
        model.load_state_dict(sd["state"])
        target = sd["target"]
        log(f"loaded {ckpt}  target={target}")
    else:
        model = train(cfg, a.train_data, a.val_data, seed=a.seed, epochs=a.epochs, bs=a.madt_bs,
                      lr=a.madt_lr, d=a.d, layers=a.layers, heads=a.heads, rtg_scale=a.rtg_scale,
                      n_cap=(60 if a.smoke else None), log=log)
        best = (-1e9, None)
        for t in a.targets:                                   # selection on the monitor seeds only
            r, o = score(model, t, mon, cfg)
            log(f"    target {t:6.1f} -> monitor {r:7.2f}  outage {o:.4f}")
            if r > best[0]:
                best = (r, t)
        target = best[1]
        log(f"selected target {target} by monitor return {best[0]:.2f}")
        torch.save(dict(state=model.state_dict(), opts=opts, target=target, layers=a.layers,
                        heads=a.heads, rtg_scale=a.rtg_scale, kind="madt"), ckpt)
        json.dump(dict(name=a.name, seed=a.seed, opts=opts, monitor_return_best=best[0],
                       madt=dict(target=target, targets=a.targets, d=a.d, layers=a.layers,
                                 heads=a.heads, epochs=a.epochs, bs=a.madt_bs, lr=a.madt_lr)),
                  open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
        log(f"saved {ckpt}")

    if a.final or a.eval_only:
        from final_eval import refs
        from planner_check import CONFIGS
        seeds = list(range(a.seed0, a.seed0 + a.n_final))
        stay, exp = refs(a.seed0, a.n_final, a.device,
                         os.path.join(a.out, "final", f"refs_{a.seed0}_{a.n_final}_{a.planner}.npz"),
                         dict(CONFIGS[a.planner], device=a.device))
        den = float(exp.mean() - stay.mean())
        r, o = score(model, target, seeds, cfg)
        log(f"[final] seeds {seeds[0]}..{seeds[-1]} (n={a.n_final}, {a.planner}): return {r:.2f}  "
            f"norm {(r - stay.mean()) / den:.3f}  outage {o:.4f}  "
            f"(stay {stay.mean():.2f}, den {den:.2f})")


if __name__ == "__main__":
    main()