"""
confirm.py -- re-evaluate every saved policy on a FRESH seed set (500000..500255) that no design decision ever looked at.
(Design choices -- energy price, BC epochs/lr -- were made while looking at returns on seeds 300000..300127, which are
inside the main 256-scenario evaluation set; this is the clean check.)
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commenv import EnvConfig, StayPut, RandomPolicy
from models import Actor, NetPolicy
from evalproto import returns, bootstrap_ci
from expert import plan, replay_returns
torch.set_num_threads(2)
CFG = EnvConfig().with_k(6.0); SEEDS = list(range(500000, 500256)); RES = "results"
stay, _ = returns(StayPut(), SEEDS, CFG); rnd, _ = returns(RandomPolicy(0), SEEDS, CFG)
o = plan(SEEDS, CFG, restarts=4, rounds=4); exp_ = replay_returns(SEEDS, CFG, o["actions"], o["channels"])
ld = lambda f: (lambda a: (a.load_state_dict(torch.load(f)), a)[1])(Actor(CFG))
lines = [f"fresh seeds 500000..500255: stay-put {stay.mean():.2f}  random {rnd.mean():.2f}  expert {exp_.mean():.2f}"]
ns = lambda r: float(np.mean((r - stay) / (exp_ - stay)))
res = {}
res["bc_only"] = [returns(NetPolicy(ld(f"{RES}/bc_seed{s}.pt")), SEEDS, CFG)[0] for s in range(5)]
for arm in ("bc_masked", "bc_unmasked", "scratch_masked", "scratch_unmasked"):
    res[arm] = [returns(NetPolicy(ld(f"{RES}/actor_{arm}_seed{s}.pt")), SEEDS, CFG)[0] for s in range(5)]
m = {a: np.array([r.mean() for r in v]) for a, v in res.items()}
for a, v in res.items():
    lo, hi = bootstrap_ci(m[a]); lines.append(f"{a:18s} mean {m[a].mean():7.2f}  95% CI over seeds [{lo:.2f}, {hi:.2f}]  expert-normalised {np.mean([ns(r) for r in v]):+.3f}")
for a, b in (("bc_masked", "bc_unmasked"), ("scratch_masked", "scratch_unmasked"), ("bc_masked", "scratch_masked"), ("bc_masked", "bc_only"), ("bc_unmasked", "bc_only")):
    d = m[a] - m[b]; lo, hi = bootstrap_ci(d, n_boot=10000)
    lines.append(f"  {a} - {b}: {d.mean():+.2f}  CI [{lo:+.2f}, {hi:+.2f}]  per-seed {np.round(d,2).tolist()}")
open(f"{RES}/confirm_fresh_seeds.txt", "w").write("\n".join(lines)); print("\n".join(lines))
