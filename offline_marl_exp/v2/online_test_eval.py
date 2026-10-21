"""
online_test_eval.py -- score the SAVED online/scratch actors on a chosen seed set, without re-running PPO.

    python online_test_eval.py --seed0 900000 --n 1024 --planner strong++        # the frozen test set
    python online_test_eval.py --seed0 800000 --n 256  --planner strong++        # dev, to reproduce the existing table
    python online_test_eval.py --tags online_A7 scratch_4500 --seed0 900000 --n 1024

Sections 2 and 3 of the v4 doc are dev-set only: online2.py wrote its `--final` returns on 800000..800255 while the runs
were going. The actors themselves were saved (`results2/online/<tag>/actor_<arm>_seed<s>.pt`), so the same policies can be
scored on any seed set after the fact -- which is what this does. Nothing is retrained.

Architecture: a tag that was produced with `--scratch` has mdn_k=0, chunk=1 (online2.py rebuilds the actor with
make_student on those options); every other tag keeps the start student's architecture. --ckpt_map says which student
checkpoint each tag started from; the defaults cover the tags in the v4 doc.
NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import torch

from commenv import EnvConfig, StayPut
from evalproto import returns, seed_level_ci, paired_diff
from models import NetPolicy
from bc2 import load_student, make_student, cfg_from_opts

# tag prefix -> (student checkpoint template, was it launched with --scratch)
DEFAULT_MAP = [
    ("online_C_b1", "results2/C_b1/seed{s}/best.pt", False),
    ("online_A7",   "results2/A7/seed{s}/best.pt",   False),
    ("scratch_",    "results2/A7/seed{s}/best.pt",   True),
    ("sw_",         "results2/A7/seed{s}/best.pt",   True),
]


def resolve(tag, extra):
    for pref, tpl, scr in extra + DEFAULT_MAP:
        if tag.startswith(pref):
            return tpl, scr
    raise SystemExit(f"no checkpoint mapping for tag {tag}; pass --ckpt_map {tag}=path/seed{{s}}/best.pt[,scratch]")


def build(ckpt, scratch):
    if scratch:
        opts = dict(torch.load(ckpt, weights_only=False)["opts"], mdn_k=0, chunk=1)
        return make_student(opts), cfg_from_opts(opts)
    return load_student(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results2/online")
    ap.add_argument("--tags", nargs="*", default=None, help="default: every tag under --root that has actor files")
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--planner", default="strong++")
    ap.add_argument("--ckpt_map", nargs="*", default=[], help="tag=template[,scratch]")
    ap.add_argument("--start_names", nargs="*", default=["A7", "C_b1"], help="also score the START students, for the gain")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from final_eval import refs
    from planner_check import CONFIGS
    extra = []
    for m in a.ckpt_map:
        tag, rest = m.split("=", 1)
        parts = rest.split(",")
        extra.append((tag, parts[0], len(parts) > 1 and parts[1] == "scratch"))

    seeds = list(range(a.seed0, a.seed0 + a.n))
    stay, exp = refs(a.seed0, a.n, a.device,
                     os.path.join("results2", "final", f"refs_{a.seed0}_{a.n}_{a.planner}.npz"),
                     dict(CONFIGS[a.planner], device=a.device))
    den = float(exp.mean() - stay.mean())
    print(f"seeds {seeds[0]}..{seeds[-1]} (n={a.n}), planner {a.planner}: stay {stay.mean():.2f} expert {exp.mean():.2f} "
          f"denominator {den:.2f}")

    tags = a.tags or sorted({os.path.basename(os.path.dirname(p))
                             for p in glob.glob(os.path.join(a.root, "*", "actor_*.pt"))})
    if not tags:
        raise SystemExit(f"no actor_*.pt under {a.root} -- fetch them from the instance (fetch.sh WITHOUT WITH_CKPT=0)")

    res = {}
    starts = {}
    for name in a.start_names:                                   # start students, so the online gain is on this seed set too
        v = []
        for p in sorted(glob.glob(os.path.join("results2", name, "seed*", "best.pt"))):
            m, cfg = load_student(p)
            v.append(float(returns(NetPolicy(m.cpu().eval()), seeds, cfg)[0].mean()))
        if v:
            starts[name] = v
            print(f"start {name:10s} {np.round(v, 2)}  mean {np.mean(v):7.2f}  norm {(np.mean(v) - stay.mean()) / den:.3f}")

    print(f"\n{'tag':18s} {'arm':9s} {'n':>2s}  per-seed return                mean     norm   95% CI over seeds")
    for tag in tags:
        tpl, scr = resolve(tag, extra)
        for arm in ("masked", "unmasked"):
            vals, ss = [], []
            for p in sorted(glob.glob(os.path.join(a.root, tag, f"actor_{arm}_seed*.pt"))):
                s = int(re.search(r"seed(\d+)\.pt$", p).group(1))
                ck = tpl.format(s=s)
                if not os.path.exists(ck):
                    print(f"  [skip] {p}: missing {ck}"); continue
                actor, cfg = build(ck, scr)
                sd = torch.load(p, map_location="cpu", weights_only=False)
                actor.load_state_dict(sd.get("state", sd) if isinstance(sd, dict) else sd)
                vals.append(float(returns(NetPolicy(actor.cpu().eval()), seeds, cfg)[0].mean())); ss.append(s)
            if not vals:
                continue
            m, ci = seed_level_ci(vals)
            res[f"{tag}/{arm}"] = dict(seeds=ss, returns=vals, mean=float(np.mean(vals)))
            print(f"{tag:18s} {arm:9s} {len(vals):2d}  {str(np.round(vals, 2)):28s} {np.mean(vals):7.2f}  "
                  f"{(np.mean(vals) - stay.mean()) / den:6.3f}  [{ci[0]:.2f}, {ci[1]:.2f}]")

    # gains against the matching start student, paired by seed where possible
    print("\ngain over the start student (same seeds, same seed set):")
    for key, d in sorted(res.items()):
        tag = key.split("/")[0]
        name = "C_b1" if tag.startswith("online_C_b1") else "A7"
        if name not in starts or tag.startswith(("scratch_", "sw_")):
            continue
        base = [starts[name][s] for s in d["seeds"] if s < len(starts[name])]
        if len(base) != len(d["returns"]):
            continue
        g = np.array(d["returns"]) - np.array(base)
        lo, hi = seed_level_ci(list(g))[1]
        print(f"  {key:28s} {g.mean():+7.2f} returns ({g.mean() / den:+.3f} norm)  per seed {np.round(g, 2)}  95% CI [{lo:.2f}, {hi:.2f}]")

    if a.out:
        json.dump(dict(seed0=a.seed0, n=a.n, planner=a.planner, stay=float(stay.mean()), expert=float(exp.mean()),
                       starts=starts, results=res), open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
