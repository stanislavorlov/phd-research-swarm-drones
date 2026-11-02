"""patch_envsize.py -- expose the environment size on the command line, and guard against silent mismatches.

    cd ~/v2 && python3 patch_envsize.py            # apply (writes .bak_envsize backups)
    cd ~/v2 && python3 patch_envsize.py --check    # verify every anchor, write nothing

EnvConfig already has n_agents / n_users / n_channels / world / alpha, but nothing exposed them, so the
pipeline could only ever run 8,32,3,20,2.5. This adds the five flags to bc2.add_student_args, expert.py and
final_eval.py, and -- the part that matters -- records them in `opts`, so a checkpoint carries the environment
it was trained in. Without that, load_student() would silently rebuild a 8,6,28 student at 8,3,20.

Every default equals the current hard-coded value, and cfg_from_opts falls back to it for checkpoints that
predate the change, so existing results reproduce bit-identically. final_eval's reference cache keeps its old
filename for the default environment, so cached references are not invalidated.

Also adds bc2.assert_env_match, called where a dataset or checkpoint meets a config, so a mismatch is a loud
failure instead of a plausible wrong number.

NOTE: written without being executed by its author. Run --check first.
"""
from __future__ import annotations
import argparse, os, py_compile, shutil, sys, tempfile

ENV_ARGS = '''    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--n_users", type=int, default=32)
    ap.add_argument("--n_channels", type=int, default=3)
    ap.add_argument("--world", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=2.5)
'''

HELPER = '''

ENV_KEYS = ("n_agents", "n_users", "n_channels", "world", "alpha", "T", "H", "noise", "energy", "vmax")


def assert_env_match(cfg, other, where=""):
    """Fail loudly when a dataset or checkpoint was produced under a different environment.

    `other` is an EnvConfig.__dict__ (expert.py stores one in every dataset) or None for artefacts that predate
    the environment-size flags, which are the default environment by construction. Observation options (k, k_a,
    bcast) are deliberately excluded: those differ between arms on purpose."""
    ref = {k: getattr(cfg, k) for k in ENV_KEYS}
    if other is None:
        other = {k: getattr(EnvConfig(), k) for k in ENV_KEYS}
    bad = {k: (ref[k], other[k]) for k in ENV_KEYS if k in other and ref[k] != other[k]}
    if bad:
        raise SystemExit("environment mismatch" + (f" for {where}" if where else "") + ": "
                         + ", ".join(f"{k}: this run has {v[0]}, the artefact has {v[1]}"
                                     for k, v in sorted(bad.items())))
'''

EDITS = {
"bc2.py": [
    # 1. the five flags, right after the observation radius
    ('    ap.add_argument("--k", type=float, default=6.0)\n',
     '    ap.add_argument("--k", type=float, default=6.0)\n' + ENV_ARGS),
    # 2. record them in opts, so every checkpoint carries its environment
    ('''def student_opts(a):
    return dict(k=a.k, d=a.d, mdn_k=(a.mdn_k if a.head == "mdn" else 0), chunk=a.chunk, obs_extra=bool(a.obs_extra),
                k_a=a.k_a, bcast=a.bcast)''',
     '''def student_opts(a):
    return dict(k=a.k, d=a.d, mdn_k=(a.mdn_k if a.head == "mdn" else 0), chunk=a.chunk, obs_extra=bool(a.obs_extra),
                k_a=a.k_a, bcast=a.bcast,
                n_agents=a.n_agents, n_users=a.n_users, n_channels=a.n_channels, world=a.world, alpha=a.alpha)'''),
    # 3. read them back, defaulting to the pre-change environment
    ('''def cfg_from_opts(opts) -> EnvConfig:
    """Env config matching a student's observation layout (old checkpoints lack k_a / bcast: defaults apply)."""
    return EnvConfig(k=opts["k"], obs_extra=opts["obs_extra"], k_a=opts.get("k_a", -1.0), bcast=opts.get("bcast", 0))''',
     '''def cfg_from_opts(opts) -> EnvConfig:
    """Env config matching a student's observation layout. Old checkpoints lack k_a / bcast and the environment
    size; the defaults below are the values those runs were produced with, so they reproduce exactly."""
    return EnvConfig(k=opts["k"], obs_extra=opts["obs_extra"], k_a=opts.get("k_a", -1.0), bcast=opts.get("bcast", 0),
                     n_agents=opts.get("n_agents", 8), n_users=opts.get("n_users", 32),
                     n_channels=opts.get("n_channels", 3), world=opts.get("world", 20.0),
                     alpha=opts.get("alpha", 2.5))''' + HELPER),
],
"expert.py": [
    ('''    ap.add_argument("--adam_iters", type=int, default=80)
    a = ap.parse_args()
    cfg = EnvConfig()''',
     '''    ap.add_argument("--adam_iters", type=int, default=80)
''' + ENV_ARGS + '''    a = ap.parse_args()
    cfg = EnvConfig(n_agents=a.n_agents, n_users=a.n_users, n_channels=a.n_channels,
                    world=a.world, alpha=a.alpha)
    print(f"environment: N={cfg.n_agents} U={cfg.n_users} C={cfg.n_channels} "
          f"W={cfg.world:g} alpha={cfg.alpha:g}", flush=True)'''),
],
"student.py": [
    ('''        d = torch.load(path, weights_only=False)''',
     '''        d = torch.load(path, weights_only=False)
        bc2.assert_env_match(cfg, d.get("cfg"), path)'''),
],
"final_eval.py": [
    ('from commenv import EnvConfig, StayPut',
     'from commenv import EnvConfig, StayPut\nfrom bc2 import assert_env_match'),
    ('def refs(seed0, n, device, path, plan_kw=None):',
     'def refs(seed0, n, device, path, plan_kw=None, cfg=None):'),
    ('    seeds = list(range(seed0, seed0 + n)); cfg = EnvConfig()',
     '    seeds = list(range(seed0, seed0 + n)); cfg = cfg or EnvConfig()'),
    ('    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")',
     ENV_ARGS + '    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")'),
    ('''    stay, exp = refs(a.seed0, a.n, a.device,
                     os.path.join(a.root, "final", f"refs_{a.seed0}_{a.n}_{a.planner}.npz"), plan_kw)''',
     '''    cfg_ref = EnvConfig(n_agents=a.n_agents, n_users=a.n_users, n_channels=a.n_channels,
                        world=a.world, alpha=a.alpha)
    # the default environment keeps the old cache filename, so existing reference caches stay valid
    env_tag = ("" if (a.n_agents, a.n_users, a.n_channels, a.world, a.alpha) == (8, 32, 3, 20.0, 2.5)
               else f"_{a.n_agents}x{a.n_users}x{a.n_channels}w{a.world:g}a{a.alpha:g}")
    stay, exp = refs(a.seed0, a.n, a.device,
                     os.path.join(a.root, "final", f"refs_{a.seed0}_{a.n}_{a.planner}{env_tag}.npz"),
                     plan_kw, cfg_ref)'''),
    ('            model, cfg = load_student(p)',
     '''            model, cfg = load_student(p)
            assert_env_match(cfg_ref, cfg.__dict__, p)'''),
],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify every anchor, write nothing")
    a = ap.parse_args()

    staged, problems = {}, []
    for fname, edits in EDITS.items():
        if not os.path.exists(fname):
            problems.append(f"{fname}: missing (run this from the v2 directory)"); continue
        src = open(fname, encoding="utf-8").read()
        for i, (old, new) in enumerate(edits, 1):
            n = src.count(old)
            if n == 0:
                if new.split("\n")[0].strip() and new.split("\n")[0] in src:
                    problems.append(f"{fname} edit {i}: already applied")
                else:
                    problems.append(f"{fname} edit {i}: anchor not found -> {old.splitlines()[0][:70]!r}")
                continue
            if n > 1:
                problems.append(f"{fname} edit {i}: anchor appears {n} times, not unique"); continue
            src = src.replace(old, new, 1)
        staged[fname] = src

    for fname, src in staged.items():                       # syntax-check the patched text before writing
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(src); tmp = fh.name
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as e:
            problems.append(f"{fname}: patched text does not compile: {e}")
        finally:
            os.unlink(tmp)

    if problems:
        print("ABORTED, nothing written:"); [print("  -", p) for p in problems]
        sys.exit(1)
    if a.check:
        print("all anchors matched and the patched text compiles; re-run without --check to apply"); return

    for fname, src in staged.items():
        shutil.copy2(fname, fname + ".bak_envsize")
        open(fname, "w", encoding="utf-8").write(src)
        print("patched", fname, "(backup:", fname + ".bak_envsize)")
    print("""
done. Validate BEFORE generating any new data -- the explicit baseline must reproduce the implicit one:

  python3 expert.py --n 20 --start 100000 --out data/_check.pt \\
      --n_agents 8 --n_users 32 --n_channels 3 --world 20 --alpha 2.5
  python3 -c "
import torch
a=torch.load('data/expert_train.pt',weights_only=False)['expert_return'][:20]
b=torch.load('data/_check.pt',weights_only=False)['expert_return']
import numpy as np; print('max abs diff', float(np.abs(a-b).max()))"

A difference of ~0 means the threading is correct. Anything larger means the explicit path differs from the
implicit one and nothing downstream should be trusted.""")


if __name__ == "__main__":
    main()