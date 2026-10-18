"""
online2.py -- MAPPO fine-tuning of the new student (any student.py checkpoint) with a k-masked vs unmasked critic.

    python online2.py --ckpt results2/A4/seed{s}/best.pt --seeds 0 1 2 3 4 --iters 150 --tag A4
    python online2.py --ckpt results2/A4/seed{s}/best.pt --seeds 0 1 2 3 4 --tag A4 --final   # once, at the very end

`{s}` in --ckpt is replaced by the seed, so student seed s is fine-tuned with PPO seed s (paired across critic arms: same
actor init, same critic init, same data seeds; the ONLY difference is the attention mask of the critic).
Eval seeds during training: 700000.. (monitor). `--final` additionally scores the last actor on 800000..800255.
Outputs results2/online/<tag>/run_<arm>_seed<s>.npz and a paired summary (CI over training seeds).
NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch

from commenv import StayPut
from evalproto import returns, seed_level_ci, bootstrap_ci
from models import NetPolicy
from mappo import PPOConfig, train_mappo
from bc2 import load_student, make_student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student checkpoint; may contain {s}")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*", default=["masked", "unmasked"])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--lr_actor", type=float, default=1e-3)
    ap.add_argument("--critic_d", type=int, default=32)
    ap.add_argument("--scratch", action="store_true", help="random-init actor (same architecture as the checkpoint, plain Gaussian head); the checkpoint is only used for its options")
    ap.add_argument("--ppo_offset", type=int, default=0, help="PPO run seed = student seed + this (keep student seed + offset <= 9)")
    ap.add_argument("--init_std", type=float, default=0.2, help="exploration std of the Gaussian around the student mean")
    ap.add_argument("--critic_warmup", type=int, default=15, help="critic-only iterations before the actor is updated")
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--tag", default="online")
    ap.add_argument("--out", default="results2/online")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n_final", type=int, default=256)
    a = ap.parse_args()
    out = os.path.join(a.out, a.tag); os.makedirs(out, exist_ok=True)
    ev = list(range(700_000, 700_000 + a.n_eval))
    fin = list(range(800_000, 800_000 + a.n_final))
    ppo = PPOConfig(iters=a.iters, n_envs=a.n_envs, lr_actor=a.lr_actor, critic_d=a.critic_d,
                    critic_warmup=a.critic_warmup, target_kl=a.target_kl)
    res = {arm: [] for arm in a.arms}
    for s in a.seeds:
        for arm in a.arms:
            f = os.path.join(out, f"run_{arm}_seed{s}.npz")
            actor, cfg = load_student(a.ckpt.format(s=s))
            if a.scratch:                                            # no offline pre-training: PPO from random weights
                opts = dict(torch.load(a.ckpt.format(s=s), weights_only=False)["opts"], mdn_k=0, chunk=1)
                torch.manual_seed(7000 + s)
                actor = make_student(opts)
            actor.log_std.data.fill_(float(np.log(a.init_std)))             # same exploration scale for every arm
            if os.path.exists(f):
                d = np.load(f); res[arm].append(float(d["returns"][-1].mean()))
                print(f"skip {f} (exists)"); continue
            with open(os.path.join(out, f"log_{arm}_seed{s}.txt"), "w") as lg:
                hist, diag, _ = train_mappo(cfg, actor, arm == "masked", s + a.ppo_offset, ppo, ev, label=f"{a.tag}-{arm} s{s}p{a.ppo_offset}", log=lg)
            its = sorted(hist)
            extra = {}
            if a.final:
                extra["final_returns"] = returns(NetPolicy(actor.cpu().eval()), fin, cfg)[0]
            np.savez(f, iters=np.array(its), returns=np.stack([hist[i] for i in its]),
                     kl=np.array([d["kl"] if d["kl"] is not None else np.nan for d in diag], dtype=float),
                     train_return=np.array([d["train_return"] for d in diag]), **extra)
            torch.save(actor.state_dict(), os.path.join(out, f"actor_{arm}_seed{s}.pt"))
            res[arm].append(float(hist[its[-1]].mean()))
    print("\nfinal-iteration eval return (mean over eval seeds), per training seed:")
    for arm, v in res.items():
        m, ci = seed_level_ci(v)
        print(f"  {arm:9s} {np.round(v, 2)}  mean {m:.2f}  95% CI over seeds {np.round(ci, 2)}")
    if len(res) == 2 and all(len(v) == len(a.seeds) for v in res.values()):
        d = np.array(res[a.arms[0]]) - np.array(res[a.arms[1]])
        print(f"  paired {a.arms[0]} - {a.arms[1]}: {d.mean():+.3f}  95% CI {np.round(bootstrap_ci(d), 3)}  (n={len(d)} seeds)")
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
