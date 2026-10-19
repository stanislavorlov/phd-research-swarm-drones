"""
v4_probes2.py -- corrected / extended version of the reviewer's v4_probes.py.

    python v4_probes2.py --ckpts results2/A7/seed0/best.pt results2/C_b1/seed0/best.pt --seed0 800000 --n 128
    python v4_probes2.py --ckpts ... --actors results2/online/online_A7/actor_masked_seed0.pt --seed0 800000 --n 128
    python v4_probes2.py --selftest                    # 2 scenarios, checks the enumeration against team_reward
    python v4_probes2.py --ub 32 --ub_delta 0.5        # certified ceiling only

What changed vs v4_probes.py, and why
  1. switch rate per step.      The initial channel assignment is uniform random over C, so a policy that re-picks ONCE at
                                t=1 already shows (1-1/C)/T = 0.033 switches per agent-step at C=3, T=20 -- the same value
                                reported as "2.5-3x more channel agility". profile_switches() reports t=1 and t>=2 apart,
                                so "re-assign once" and "keep re-tuning" stop being the same number.
  2. exact team headroom.       v4_probes measured a UNILATERAL best response (one UAV deviates, others fixed) and then
                                attributed the remaining A7->C_b1 gap to trajectory shape. Unilateral gains are not a bound
                                on what joint re-selection can do. headroom() enumerates all C^N joint assignments (6561 at
                                N=8, C=3) at every visited state and reports the exact per-state optimum alongside the
                                unilateral number, so the two can be compared.
  3. causal rule inputs.        The channel rules are scored from the state BEFORE the step (what a receiver could actually
                                have measured), and separately from the post-step state (the reviewer's version) -- the
                                difference is the value of knowing the others' simultaneous choices, not of sensing.
  4. tie handling.              argmin over an all-zero sensed vector always returns channel 0, and the best response is
                                often tied (UAVs serving no user). A pick counts as correct if it is in the argmax set
                                (tolerance --tol), and the table is reported for all UAVs and for serving UAVs only.
  5. more behaviour columns.    mean rate, rate quantiles, mean distance user->serving UAV, dispersion of the UAV swarm.
                                Outage (rate < 1) is a tail statistic; it moves for B_ainf/A8_kinf but barely for C_b1,
                                which nevertheless gains ~26 returns, so the mean has to be in the table too.
  6. design sizing.             Two variance models: the reviewer's two-way one (PPO offset treated as a crossed factor)
                                and a pooled one (students as the only unit). The offsets are NOT shared across students
                                (run seed = student seed + offset), so the pooled model is the defensible one; both are
                                printed because they disagree by ~0.2 returns of half-width.
  7. certified bound.           Same construction (valid: energy >= 0 dropped, speed limit dropped, interference zeroed,
                                each step optimised independently, gridded k-median LP + Lipschitz slack) but the slack
                                is reported separately -- it is 0.63/step at delta=1 (12.7 over an episode), so delta=0.5
                                is the setting to quote.

Scores go on the DEV seeds (800000+) by default; keep the frozen test set (900000+) out of every diagnostic.
NOTE: written without being executed by its author, except --selftest (2 scenarios, correctness only).
"""
from __future__ import annotations
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from commenv import CommEnv, EnvConfig, StayPut, link_rates, team_reward
from models import NetPolicy, StudentActor


# ----------------------------------------------------------------------------- loading
def load_any(path):
    """A student checkpoint (dict with opts/state) or a bare online actor state_dict (needs --cfg_from)."""
    c = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(c, dict) and "opts" in c:
        o = c["opts"]
        cfg = EnvConfig(k=o["k"], k_a=o.get("k_a", -1.0), bcast=o.get("bcast", 0), obs_extra=o.get("obs_extra", False))
        a = StudentActor(cfg, d=o["d"], mdn_k=o.get("mdn_k", 0), chunk=o.get("chunk", 1))
        a.load_state_dict(c["state"]); a.eval()
        return a, cfg
    raise ValueError(f"{path} is a bare state_dict; pass --cfg_from <student ckpt> so the architecture is known")


def load_actor_with_cfg(path, cfg_from):
    """Online-tuned actor (online2.py saves actor.state_dict()); architecture taken from the student it started from."""
    base, cfg = load_any(cfg_from)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    base.load_state_dict(sd if not isinstance(sd, dict) or "state" not in sd else sd["state"])
    base.eval()
    return base, cfg


# ----------------------------------------------------------------------------- behaviour
@torch.no_grad()
def behaviour(actor, cfg, seeds, name="policy", near=6.0):
    """What the policy DOES. Switch rate is reported per step as well as pooled (see profile_switches)."""
    env = CommEnv(cfg, seeds); obs = env.reset()
    pol = NetPolicy(actor) if actor is not None else StayPut()
    N = cfg.n_agents
    mv, rate_t, out_t, co_t, sw_t, serv_d, disp_t = [], [], [], [], [], [], []
    ret = torch.zeros(env.B); prev = env.ch.clone()
    rates_all = []
    while True:
        disp, ch = pol.act(env, obs)
        d = disp.norm(dim=-1, keepdim=True)
        mv.append((disp / d.clamp(min=1.0)).norm(dim=-1).flatten())          # the executed (clipped) move
        sw_t.append(float((ch != prev).float().mean())); prev = ch.clone()
        obs, r, done, info = env.step(disp, ch)
        ret += r
        users = env.users_traj[:, env.t]
        rr, serv = link_rates(env.uav, users, env.ch, cfg)
        rates_all.append(rr.flatten())
        rate_t.append(float(rr.mean())); out_t.append(float((rr < 1.0).float().mean()))
        sd = (users - env.uav.gather(1, serv.unsqueeze(-1).expand(-1, -1, 2))).norm(dim=-1)
        serv_d.append(float(sd.mean()))
        dd = ((env.uav.unsqueeze(2) - env.uav.unsqueeze(1)) ** 2).sum(-1).sqrt()
        same = (env.ch.unsqueeze(2) == env.ch.unsqueeze(1)) & ~torch.eye(N, dtype=torch.bool)
        co_t.append(float(((dd < near) & same).float().sum((1, 2)).mean() / 2))
        disp_t.append(env.uav.std(dim=1).mean().item())
        if done:
            q = torch.quantile(torch.cat(rates_all), torch.tensor([0.1, 0.5, 0.9]))
            return dict(name=name, ret=float(ret.mean()), ret_sd=float(ret.std()),
                        move=float(torch.cat(mv).mean()),
                        switch_all=float(np.mean(sw_t)), switch_t1=sw_t[0], switch_t2plus=float(np.mean(sw_t[1:])),
                        rate=float(np.mean(rate_t)), rate_q10=float(q[0]), rate_q50=float(q[1]), rate_q90=float(q[2]),
                        outage=float(np.mean(out_t)), cochan=float(np.mean(co_t)),
                        serv_dist=float(np.mean(serv_d)), swarm_spread=float(np.mean(disp_t)),
                        switch_profile=[round(x, 4) for x in sw_t])


# ----------------------------------------------------------------------------- channel probes
def _rate_given(g, serv, S, ch):
    """rate [B,U] for assignment ch [B,N], with the geometry (g, serv, S) precomputed. Serving UAV does not depend on ch."""
    cfg_noise = _rate_given.noise
    ch_s = ch.gather(-1, serv)                                               # [B,U]
    same = (ch.unsqueeze(-2) == ch_s.unsqueeze(-1)).to(g.dtype)              # [B,U,N]
    I = (g * same).sum(-1) - S
    return torch.log2(1.0 + S / (cfg_noise + I))


@torch.no_grad()
def joint_optimum(uav, users, ch, cfg, chunk=512):
    """Exact max over ALL C^N joint channel assignments. Returns (J_now [B], J_opt [B], J_uni [B]).
    J_uni = best single-UAV unilateral deviation (the quantity v4_probes reported)."""
    B, N, C = uav.shape[0], cfg.n_agents, cfg.n_channels
    d2 = ((users.unsqueeze(-2) - uav.unsqueeze(-3)) ** 2).sum(-1)            # [B,U,N]
    g = (d2 + cfg.H ** 2) ** (-cfg.alpha / 2.0)
    serv = g.argmax(-1)
    S = g.gather(-1, serv.unsqueeze(-1)).squeeze(-1)
    _rate_given.noise = cfg.noise
    J_now = _rate_given(g, serv, S, ch).mean(-1)
    # unilateral
    J_uni = J_now.clone()
    for i in range(N):
        for c in range(C):
            ch2 = ch.clone(); ch2[:, i] = c
            J_uni = torch.maximum(J_uni, _rate_given(g, serv, S, ch2).mean(-1))
    # joint
    total = C ** N
    if total > 500_000:
        return J_now, torch.full_like(J_now, float("nan")), J_uni
    J_opt = torch.full_like(J_now, -1e9)
    base = torch.arange(total)
    for s in range(0, total, chunk):
        idx = base[s:s + chunk]
        A = torch.stack([(idx // (C ** j)) % C for j in range(N)], -1)       # [a,N]
        chA = A.unsqueeze(1).expand(-1, B, -1)                               # [a,B,N]
        Ja = _rate_given(g.unsqueeze(0), serv.unsqueeze(0).expand(chA.shape[0], -1, -1),
                         S.unsqueeze(0), chA).mean(-1)                       # [a,B]
        J_opt = torch.maximum(J_opt, Ja.max(0).values)
    return J_now, J_opt, J_uni


@torch.no_grad()
def channel_rules(actor, cfg, seeds, k_vis=6.0, tol=1e-6, exact=True, noise_db=0.0, rng_seed=0):
    """
    At every visited state compare channel-selection rules against the exact best response.
    Rules are scored twice:
      causal  -- the rule sees the state BEFORE the step (positions/channels of the previous step),
      oracle  -- the rule sees the post-step state (what v4_probes did; needs the others' simultaneous choices).
    A pick counts as correct if it is in the argmax set (ties are common: a UAV serving no user is indifferent).
    `noise_db` adds log-normal measurement noise to the sensed power (realism check; 0 = perfect sensing).
    """
    env = CommEnv(cfg, seeds); obs = env.reset(); pol = NetPolicy(actor)
    B, N, C = env.B, cfg.n_agents, cfg.n_channels
    g_rng = torch.Generator().manual_seed(rng_seed)
    keys = ("sensed", "sensed_k", "bcast", "stay", "rand")
    acc = {m: {k: [] for k in keys} for m in ("causal", "oracle")}
    acc_srv = {m: {k: [] for k in keys} for m in ("causal", "oracle")}
    gaps = dict(uni=[], opt=[], now=[])
    eye = torch.eye(N, dtype=torch.bool)

    def feats(pos, chc):
        d2 = ((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2).sum(-1)
        gg = ((d2 + cfg.H ** 2) ** (-cfg.alpha / 2.0)).masked_fill(eye, 0.0)
        oh = torch.nn.functional.one_hot(chc, C).float()
        sensed = torch.einsum("bij,bjc->bic", gg, oh)
        if noise_db > 0:
            sensed = sensed * torch.exp(torch.randn(sensed.shape, generator=g_rng) * (noise_db / 8.686))
        vis = (d2.sqrt() <= k_vis) & ~eye
        sensed_k = torch.einsum("bij,bjc->bic", gg * vis.float(), oh)
        count = torch.einsum("bij,bjc->bic", (~eye).float().expand(B, N, N), oh)
        return sensed, sensed_k, count

    while True:
        pre_pos, pre_ch = env.uav.clone(), env.ch.clone()
        pre = feats(pre_pos, pre_ch)
        disp, ch = pol.act(env, obs)
        obs, r, done, info = env.step(disp, ch)
        pos, users, chc = env.uav, env.users_traj[:, env.t], env.ch
        post = feats(pos, chc)

        d2u = ((users.unsqueeze(-2) - pos.unsqueeze(-3)) ** 2).sum(-1)
        gu = (d2u + cfg.H ** 2) ** (-cfg.alpha / 2.0)
        serv = gu.argmax(-1); Su = gu.gather(-1, serv.unsqueeze(-1)).squeeze(-1)
        _rate_given.noise = cfg.noise
        J = torch.stack([_rate_given(gu, serv, Su, torch.where(
            torch.arange(N).view(1, N) == i, c, chc)).mean(-1) for i in range(N) for c in range(C)], -1).view(B, N, C)
        Jbest = J.max(-1, keepdim=True).values
        is_best = (J >= Jbest - tol)                                          # [B,N,C] argmax SET
        serving = torch.zeros(B, N, dtype=torch.bool)
        serving.scatter_(1, serv, True)

        for mode, (sensed, sensed_k, count) in (("causal", pre), ("oracle", post)):
            picks = dict(sensed=sensed.argmin(-1), sensed_k=sensed_k.argmin(-1), bcast=count.argmin(-1),
                         stay=chc, rand=torch.randint(0, C, (B, N), generator=g_rng))
            for k, p in picks.items():
                ok = is_best.gather(-1, p.unsqueeze(-1)).squeeze(-1)
                acc[mode][k].append(ok.float().flatten())
                acc_srv[mode][k].append(ok[serving].float())

        J_now, J_opt, J_uni = joint_optimum(pos, users, chc, cfg) if exact else (None, None, None)
        if exact:
            gaps["now"].append(J_now); gaps["opt"].append(J_opt); gaps["uni"].append(J_uni)
        if done:
            f = lambda d: {k: round(float(torch.cat(v).mean()), 4) for k, v in d.items()}
            out = dict(accuracy_all={m: f(acc[m]) for m in acc}, accuracy_serving={m: f(acc_srv[m]) for m in acc_srv})
            if exact:
                now = torch.stack(gaps["now"]); opt = torch.stack(gaps["opt"]); uni = torch.stack(gaps["uni"])
                out["headroom_per_episode"] = dict(
                    joint_optimal=round(float((opt - now).sum(0).mean()), 3),
                    unilateral_best=round(float((uni - now).sum(0).mean()), 3),
                    note="per-episode sum over T of the per-state gain; 'unilateral' is the v4_probes quantity")
            return out


# ----------------------------------------------------------------------------- certified ceiling
def certified_ub(seeds, cfg, delta=0.5):
    """Upper bound on the return of ANY feasible policy. Same construction as v4_probes.certified_ub; the
    Lipschitz slack (L*delta/sqrt2 per step) is returned separately because it is a large part of the bound."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, csr_matrix, hstack, identity, vstack
    gx = np.arange(0, cfg.world + 1e-9, delta)
    G = np.stack(np.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
    F = len(G)
    v = lambda d2: np.log2(1.0 + (d2 + cfg.H ** 2) ** (-cfg.alpha / 2.0) / cfg.noise)
    dd = np.linspace(0, cfg.world * 1.5, 20001)
    L = float(np.max(np.abs(np.diff(v(dd ** 2)) / np.diff(dd))))
    slack = L * delta / np.sqrt(2.0)
    out = []
    for s in seeds:
        env = CommEnv(cfg, [s]); U = cfg.n_users; tot = 0.0
        for t in range(1, cfg.T + 1):
            X = env.users_traj[0, t].numpy()
            V = v(((X[:, None, :] - G[None, :, :]) ** 2).sum(-1))
            nx = U * F
            c = np.concatenate([-(V.reshape(-1)) / U, np.zeros(F)])
            rows = np.repeat(np.arange(U), F); cols = np.arange(nx)
            Aeq1 = coo_matrix((np.ones(nx), (rows, cols)), shape=(U, nx + F))
            Aeq2 = coo_matrix((np.ones(F), (np.zeros(F), np.arange(nx, nx + F))), shape=(1, nx + F))
            Aub = hstack([identity(nx, format="coo"),
                          coo_matrix((-np.ones(nx), (np.arange(nx), np.tile(np.arange(F), U))), shape=(nx, F))])
            r = linprog(c, A_ub=csr_matrix(Aub), b_ub=np.zeros(nx), A_eq=vstack([Aeq1, Aeq2], format="csr"),
                        b_eq=np.concatenate([np.ones(U), [cfg.n_agents]]), bounds=(0, 1), method="highs")
            assert r.status == 0, r.message
            tot += -r.fun + slack
        out.append(tot)
    return np.array(out), L, slack * cfg.T


# ----------------------------------------------------------------------------- design sizing
def design_table(d, max_s=16):
    """d[p, s] = paired contrast for replicate p, student s. Prints both variance models (see header note 6)."""
    from scipy import stats
    d = np.asarray(d, float); P, S = d.shape
    gm = d.mean(); stu = d.mean(0); off = d.mean(1)
    ms_stu = P * ((stu - gm) ** 2).sum() / (S - 1)
    ms_res = ((d - stu[None, :] - off[:, None] + gm) ** 2).sum() / ((S - 1) * (P - 1))
    two = dict(var_student=max((ms_stu - ms_res) / P, 0.0), var_resid=ms_res)
    v_stu = stu.var(ddof=1)                                                  # pooled: students are the unit
    v_res = float(np.mean(d.var(axis=0, ddof=1))) if P > 1 else 0.0
    pooled = dict(var_student=max(v_stu - v_res / P, 0.0), var_resid=v_res)
    hw = lambda S_, P_, m: float(stats.t.ppf(0.975, S_ - 1) * math.sqrt(m["var_student"] / S_ + m["var_resid"] / (S_ * P_)))
    print(f"\nmean contrast {gm:+.3f} over {S} students x {P} replicates")
    for nm, m in (("two-way (reviewer)", two), ("pooled (students as unit)", pooled)):
        print(f"  {nm:26s} var_student {m['var_student']:.3f}  var_resid {m['var_resid']:.3f}   "
              f"current {S}x{P}: +/-{hw(S, P, m):.2f}")
    print(f"\n{'students':>8s} " + " ".join(f"{'P=' + str(p):>12s}" for p in (1, 2, 3)) + "   (pooled model)")
    for s_ in range(3, max_s + 1):
        print(f"{s_:8d} " + " ".join(f"{hw(s_, p, pooled):12.2f}" for p in (1, 2, 3)))


# ----------------------------------------------------------------------------- self-test
def selftest():
    cfg = EnvConfig()
    env = CommEnv(cfg, [800000, 800001]); env.reset()
    env.step(torch.zeros(2, cfg.n_agents, 2), env.ch)
    users = env.users_traj[:, env.t]
    now, opt, uni = joint_optimum(env.uav, users, env.ch, cfg)
    ref = team_reward(env.uav, users, env.ch, cfg)
    assert torch.allclose(now, ref, atol=1e-5), (now, ref)
    assert (opt >= uni - 1e-6).all() and (uni >= now - 1e-6).all(), (now, uni, opt)
    print(f"selftest ok: J_now == team_reward (max diff {float((now - ref).abs().max()):.2e}); "
          f"now {now.mean():.3f} <= unilateral {uni.mean():.3f} <= joint {opt.mean():.3f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[], help="student checkpoints (results2/<name>/seed<s>/best.pt)")
    ap.add_argument("--actors", nargs="*", default=[], help="online-tuned actor state_dicts (need --cfg_from)")
    ap.add_argument("--cfg_from", default=None, help="student ckpt giving the architecture for --actors")
    ap.add_argument("--seed0", type=int, default=800_000, help="DEV seeds; keep 900000+ (test) out of diagnostics")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--noise_db", type=float, default=0.0, help="log-normal sensing noise, dB (realism check)")
    ap.add_argument("--no_exact", action="store_true", help="skip the C^N joint optimum")
    ap.add_argument("--ub", type=int, default=0, help="scenarios for the certified ceiling (0 = skip)")
    ap.add_argument("--ub_delta", type=float, default=0.5)
    ap.add_argument("--contrasts", default=None, help="JSON file with a [P][S] array of paired contrasts for sizing")
    ap.add_argument("--out", default=None, help="write all results as JSON here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    sds = list(range(a.seed0, a.seed0 + a.n))
    res = {}
    if a.ckpts or a.actors:
        base = EnvConfig()
        res["stay_put"] = behaviour(None, base, sds, "stay-put")
        print(json.dumps(res["stay_put"], indent=1))
        for p in a.ckpts:
            act, cfg = load_any(p)
            res[p] = behaviour(act, cfg, sds, p)
            print("\n" + json.dumps(res[p], indent=1))
            res[p + "::channels"] = channel_rules(act, cfg, sds, k_vis=cfg.k, exact=not a.no_exact, noise_db=a.noise_db)
            print("  channels:", json.dumps(res[p + "::channels"], indent=1))
        for p in a.actors:
            if not a.cfg_from:
                raise SystemExit("--actors needs --cfg_from <student ckpt>")
            act, cfg = load_actor_with_cfg(p, a.cfg_from)
            res[p] = behaviour(act, cfg, sds, p)
            print("\n" + json.dumps(res[p], indent=1))
            res[p + "::channels"] = channel_rules(act, cfg, sds, k_vis=cfg.k, exact=not a.no_exact, noise_db=a.noise_db)
            print("  channels:", json.dumps(res[p + "::channels"], indent=1))
    if a.ub:
        ub, L, ep_slack = certified_ub(range(a.seed0, a.seed0 + a.ub), EnvConfig(), delta=a.ub_delta)
        res["certified"] = dict(mean=float(ub.mean()), sd=float(ub.std()), L=L, episode_slack=ep_slack, delta=a.ub_delta)
        print(f"\ncertified ceiling {ub.mean():.2f} +/- {ub.std():.2f} (n={a.ub}, delta={a.ub_delta}); "
              f"Lipschitz {L:.3f}, of which slack {ep_slack:.2f} returns per episode")
    if a.contrasts:
        design_table(json.load(open(a.contrasts)))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
