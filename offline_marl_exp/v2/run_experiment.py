"""
run_experiment.py -- the whole study, one command per stage.

  python run_experiment.py refs                    # stay-put / random / expert / weak-expert / UB-proxy on the eval seeds
  python run_experiment.py bc     --seeds 0 1 2 3 4        # distil the planner into a k-masked set-actor (gate G3 enforced)
  python run_experiment.py online --seeds 0 1 2 3 4 --arms bc_masked bc_unmasked scratch_masked scratch_unmasked
  python run_experiment.py analyze                 # paired tables + figure

Arms (each differs from its comparison partner in exactly ONE factor):
  bc_masked        BC init  + MAPPO, k-masked critic      ("Ours")
  bc_unmasked      BC init  + MAPPO, unmasked critic      (differs from Ours: the attention mask only)
  scratch_masked   random   + MAPPO, k-masked critic      (differs from Ours: initialisation only)
  scratch_unmasked random   + MAPPO, unmasked critic
  BC-only          = checkpoint 0 of the bc_* arms (no fine-tuning)

Seed s fixes: BC init/shuffle, critic init, PPO sampling noise and the training-scenario sequence -> arms with the
same seed are paired. Evaluation is deterministic, on a held-out seed set shared by every arm and checkpoint.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commenv import EnvConfig, StayPut, RandomPolicy
from models import Actor, NetPolicy
from bc import train_bc
from mappo import PPOConfig, train_mappo
from evalproto import returns, paired_diff, expert_normalised
from expert import plan, replay_returns

K = 6.0
CFG = EnvConfig().with_k(K)
EVAL_SEEDS = list(range(300000, 300256))
RES = "results"
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(1)


def refs():
    out = {}
    out["stay_put"], _ = returns(StayPut(), EVAL_SEEDS, CFG)
    out["random"], _ = returns(RandomPolicy(0), EVAL_SEEDS, CFG)
    t = time.time(); strong = plan(EVAL_SEEDS, CFG, restarts=4, rounds=4)
    out["expert"] = replay_returns(EVAL_SEEDS, CFG, strong["actions"], strong["channels"])
    out["expert_solve_s_batched_per_scenario"] = np.array(strong["solve_s"] / len(EVAL_SEEDS))
    weak = plan(EVAL_SEEDS, CFG, restarts=1, rounds=1, adam_iters=20)
    out["weak_expert"] = replay_returns(EVAL_SEEDS, CFG, weak["actions"], weak["channels"])
    ub = plan(EVAL_SEEDS, CFG, restarts=4, rounds=4, interference_free=True)
    out["ub_proxy"] = ub["J"]
    t = time.time(); plan(EVAL_SEEDS[:1], CFG, restarts=4, rounds=4); out["expert_solve_s_single"] = np.array(time.time() - t)
    np.savez(f"{RES}/refs.npz", **out)
    for k, v in out.items():
        print(f"{k:42s} {np.mean(v):10.3f}")


def bc(seeds, epochs, lr):
    for s in seeds:
        actor, hist = train_bc(CFG, "data/expert_train.pt", "data/expert_val.pt", seed=s, epochs=epochs, lr=lr)
        torch.save(actor.state_dict(), f"{RES}/bc_seed{s}.pt")
        r, _ = returns(NetPolicy(actor), EVAL_SEEDS, CFG)
        st, _ = returns(StayPut(), EVAL_SEEDS, CFG); rn, _ = returns(RandomPolicy(0), EVAL_SEEDS, CFG)
        d1, d2 = paired_diff(r, st), paired_diff(r, rn)
        print(f"BC seed {s}: return {r.mean():.2f}  vs stay-put {d1['mean']:+.2f} CI {d1['ci95']}  vs random {d2['mean']:+.2f} CI {d2['ci95']}")
        assert d1["ci95"][0] > 0 and d2["ci95"][0] > 0, "GATE G3 FAILED: student does not beat the trivial floors"
        # latency of one decentralised decision (single scenario, all agents)
        from commenv import CommEnv
        e = CommEnv(CFG, [EVAL_SEEDS[0]]); o = e.reset(); pol = NetPolicy(actor)
        t = time.time()
        for _ in range(200): pol.act(e, o)
        print(f"   inference: {(time.time()-t)/200*1e3:.2f} ms per step for the whole swarm (1 thread, CPU)")


def madt_stage(seeds, epochs):
    from madt import train_madt, MADTPolicy
    ex_train = float(torch.load("data/expert_train.pt", weights_only=False)["expert_return"].mean())
    stay, _ = returns(StayPut(), EVAL_SEEDS, CFG); rnd, _ = returns(RandomPolicy(0), EVAL_SEEDS, CFG)
    out = {}
    for s in seeds:
        model, hist = train_madt(CFG, "data/expert_train.pt", "data/expert_val.pt", seed=s, epochs=epochs)
        torch.save(model.state_dict(), f"{RES}/madt_seed{s}.pt")
        for tgt in (ex_train, 0.5 * ex_train + 0.5 * float(stay.mean())):
            r, _ = returns(MADTPolicy(model, tgt), EVAL_SEEDS, CFG)
            d1, d2 = paired_diff(r, stay), paired_diff(r, rnd)
            print(f"MADT seed {s} (return-to-go target {tgt:.1f}): return {r.mean():.2f}  vs stay-put {d1['mean']:+.2f} CI {d1['ci95']}  vs random {d2['mean']:+.2f} CI {d2['ci95']}")
            out[f"seed{s}_target{tgt:.0f}"] = r
    np.savez(f"{RES}/madt_eval.npz", **out)


def online(seeds, arms, iters, lr_actor=1e-3):
    ppo = PPOConfig(iters=iters, lr_actor=lr_actor)
    for s in seeds:
        for arm in arms:
            f = f"{RES}/run_{arm}_seed{s}.npz"
            if os.path.exists(f):
                continue
            init, crit = arm.split("_")
            torch.manual_seed(s)
            actor = Actor(CFG)
            if init == "bc":
                actor.load_state_dict(torch.load(f"{RES}/bc_seed{s}.pt"))
            # identical exploration scale for every arm (one factor at a time)
            actor.log_std.data.fill_(float(np.log(0.2)))
            with open(f"{RES}/log_{arm}_seed{s}.txt", "w") as lg:
                hist, diag, critic = train_mappo(CFG, actor, crit == "masked", s, ppo, EVAL_SEEDS, label=f"{arm} s{s}", log=lg)
            its = sorted(hist)
            np.savez(f, iters=np.array(its), returns=np.stack([hist[i] for i in its]),
                     kl=np.array([d["kl"] if d["kl"] is not None else np.nan for d in diag], dtype=float),
                     clipfrac=np.array([d["clipfrac"] if d["clipfrac"] is not None else np.nan for d in diag], dtype=float),
                     train_return=np.array([d["train_return"] for d in diag]))
            torch.save(actor.state_dict(), f"{RES}/actor_{arm}_seed{s}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refs", "bc", "madt", "online", "analyze"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*", default=["bc_masked", "bc_unmasked", "scratch_masked", "scratch_unmasked"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr_actor", type=float, default=1e-3)
    a = ap.parse_args()
    if a.stage == "refs": refs()
    elif a.stage == "bc": bc(a.seeds, a.epochs, a.lr)
    elif a.stage == "madt": madt_stage(a.seeds, a.epochs)
    elif a.stage == "online": online(a.seeds, a.arms, a.iters, a.lr_actor)
    else:
        import analyze; analyze.main()
