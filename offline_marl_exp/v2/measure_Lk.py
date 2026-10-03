"""
measure_Lk.py -- empirical far-field influence L(k) and the bound it is compared against (review §5 step 6, Figure 2).

Definitions (see gap_bound_theorem_v2.md):
  k-local reward  r^{(k)} : same reward, but for every user the interferers whose distance to the user's
                  SERVING UAV exceeds k are dropped (they are outside what that UAV can observe).
  eps_r(k) = r^{(k)} - r >= 0   (dropping interference can only raise the rate), measured per state.
  L_hat(k) = E_s[ eps_r(k) ] over states visited by the expert, plus the 95th percentile and the max.

Bounds drawn against it (Prop. 1+2 of the theorem note, P/N0 normalised, rho = measured service radius):
  deterministic :  log2( 1 + (N-1) * gbar(k-rho) / N0 )
  density-based :  log2( 1 + [ 2*pi*lam * ((k-rho)^2+H^2)^((2-alpha)/2) / (alpha-2) ] / N0 )   (alpha > 2 needed)
where gbar(x) = (x_+^2 + H^2)^(-alpha/2) and lam = N / W^2.

Second panel: PAIRWISE influence. Switch one UAV j to another channel, measure |change| in UAV i's own local
reward, bin by d_ij, and compare the decay with the path-loss law g(d) ~ d^-alpha.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commenv import EnvConfig, CommEnv, link_rates, agent_rewards
from expert import plan

torch.set_num_threads(2)
CFG = EnvConfig()
SEEDS = list(range(400000, 400096))
K_GRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 28]


def expert_states(seeds):
    out = plan(seeds, CFG, restarts=2, rounds=2)
    env = CommEnv(CFG, seeds); env.reset()
    P, C = [], []
    for t in range(CFG.T):
        env.step(torch.tensor(out["actions"][:, t]), torch.tensor(out["channels"][:, t]).long())
        P.append(env.uav.clone()); C.append(env.ch.clone())
    return torch.stack(P, 1), torch.stack(C, 1), env.users_traj[:, 1:]     # [S,T,N,2], [S,T,N], [S,T,U,2]


def k_local_rates(pos, users, ch, k):
    g = (((users.unsqueeze(-2) - pos.unsqueeze(-3)) ** 2).sum(-1) + CFG.H ** 2) ** (-CFG.alpha / 2)   # [...,U,N]
    serv = g.argmax(-1)
    S = g.gather(-1, serv.unsqueeze(-1)).squeeze(-1)
    ch_s = ch.gather(-1, serv)
    same = (ch.unsqueeze(-2) == ch_s.unsqueeze(-1)).float()
    pos_s = torch.gather(pos, -2, serv.unsqueeze(-1).expand(*serv.shape, 2))                      # serving UAV position
    d_sj = (pos_s.unsqueeze(-2) - pos.unsqueeze(-3)).norm(dim=-1)                                   # [...,U,N]
    notself = 1.0 - torch.nn.functional.one_hot(serv, pos.shape[-2]).float()
    I_all = (g * same * notself).sum(-1)
    I_near = (g * same * notself * (d_sj <= k).float()).sum(-1)
    rate = torch.log2(1 + S / (CFG.noise + I_all))
    rate_k = torch.log2(1 + S / (CFG.noise + I_near))
    rho = (users - pos_s).norm(dim=-1)                                                              # user->serving distance
    return rate, rate_k, rho, I_all - I_near, I_near


def main():
    os.makedirs("results", exist_ok=True)
    pos, ch, users = expert_states(SEEDS)
    N, U, H, al, W, N0 = CFG.n_agents, CFG.n_users, CFG.H, CFG.alpha, CFG.world, CFG.noise
    res = {}
    rho_all = None
    for k in K_GRID:
        rate, rate_k, rho, Ifar, Inear = k_local_rates(pos, users, ch, float(k))
        eps = (rate_k - rate).mean(-1).flatten()                                  # team-level (mean over users) per state
        p1 = torch.log2(1 + Ifar / (CFG.noise + Inear)).mean(-1).flatten()      # Prop. 1 evaluated on the measured I_far
        res[k] = dict(prop1_on_measured=float(p1.mean()), mean=float(eps.mean()), p95=float(eps.quantile(0.95)), max=float(eps.max()),
                      Ifar_mean=float(Ifar.mean()), Ifar_max=float(Ifar.max()))
        rho_all = rho
    rho95 = float(rho_all.quantile(0.95)); rho_max = float(rho_all.max())
    lam = N / W ** 2
    gbar = lambda x: (max(x, 0.0) ** 2 + H ** 2) ** (-al / 2)
    for k in K_GRID:
        x = k - rho95
        res[k]["bound_det"] = float(np.log2(1 + (N - 1) * gbar(x) / N0))
        res[k]["bound_density"] = float(np.log2(1 + 2 * np.pi * lam * ((max(x, 0) ** 2 + H ** 2) ** ((2 - al) / 2)) / (al - 2) / N0))
    # ---- pairwise influence
    S_, T_ = pos.shape[:2]
    bins = np.array([0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 30])
    sums, cnts = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    base = agent_rewards(pos, users, ch, CFG) * CFG.n_users                         # [S,T,N] rate share of agent i (x U)
    dmat = (pos.unsqueeze(-2) - pos.unsqueeze(-3)).norm(dim=-1)                     # [S,T,N,N]
    for j in range(N):
        ch2 = ch.clone(); ch2[..., j] = (ch2[..., j] + 1) % CFG.n_channels
        new = agent_rewards(pos, users, ch2, CFG) * CFG.n_users
        infl = (new - base).abs()                                                   # [S,T,N] effect on every agent i
        for i in range(N):
            if i == j: continue
            d = dmat[..., i, j].flatten().numpy(); v = infl[..., i].flatten().numpy()
            idx = np.digitize(d, bins) - 1
            for b in range(len(bins) - 1):
                m = idx == b
                sums[b] += v[m].sum(); cnts[b] += m.sum()
    pair = sums / np.maximum(cnts, 1)
    mid = 0.5 * (bins[:-1] + bins[1:])
    ok = (cnts > 50) & (mid >= 6)
    slope = float(np.polyfit(np.log(mid[ok]), np.log(np.maximum(pair[ok], 1e-12)), 1)[0]) if ok.sum() >= 3 else float("nan")
    json.dump(dict(per_k=res, rho95=rho95, rho_max=rho_max, pair_bins=bins.tolist(), pair_mean=pair.tolist(),
                   pair_counts=cnts.tolist(), pair_loglog_slope_d_ge_6=slope, alpha=al), open("results/Lk.json", "w"), indent=1)
    print(f"service radius rho: 95th pct {rho95:.2f}, max {rho_max:.2f}")
    print(f"{'k':>4} {'L_hat mean':>11} {'p95':>8} {'max':>8} {'Prop1(meas)':>12} {'bound(det)':>11} {'bound(density)':>15}")
    for k in K_GRID:
        r = res[k]; print(f"{k:4d} {r['mean']:11.4f} {r['p95']:8.4f} {r['max']:8.4f} {r['prop1_on_measured']:12.4f} {r['bound_det']:11.4f} {r['bound_density']:15.4f}")
    print(f"pairwise influence log-log slope (d>=6): {slope:.2f}   (path-loss exponent alpha = {al}; far-field slope should approach -alpha)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ks = np.array(K_GRID)
    ax[0].plot(ks, [res[k]["mean"] for k in ks], "o-", c="#1f77b4", label=r"measured $\hat L(k)$ (mean over expert states)")
    ax[0].plot(ks, [res[k]["p95"] for k in ks], "o--", c="#1f77b4", alpha=.6, label="measured, 95th percentile")
    ax[0].plot(ks, [res[k]["prop1_on_measured"] for k in ks], "d-", c="#ff7f0e", label=r"Prop. 1 on the measured $I_{far}$ (per-state lemma, tight)")
    ax[0].plot(ks, [res[k]["bound_det"] for k in ks], "s-", c="#d62728", label=r"bound: $\log_2(1+(N-1)\bar g(k-\rho)/N_0)$")
    ax[0].plot(ks, [res[k]["bound_density"] for k in ks], "^-", c="#2ca02c", label=r"bound: density integral, needs $\alpha>2$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("observation radius k"); ax[0].set_ylabel("rate lost to unobserved far-field (bit/s/Hz per user)")
    ax[0].set_title("Far-field influence vs observation radius"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3, which="both")
    ok2 = cnts > 0
    ax[1].loglog(mid[ok2], pair[ok2], "o-", label="measured: |Δ own rate| when one UAV switches channel")
    ref = pair[ok][0] * (mid[ok2] / mid[ok][0]) ** (-al) if ok.any() else None
    if ref is not None: ax[1].loglog(mid[ok2], ref, "k--", label=fr"$d^{{-\alpha}}$, $\alpha$={al}")
    ax[1].set_xlabel("distance between the two UAVs"); ax[1].set_ylabel("mean |Δ| of UAV i summed user rate (bit/s/Hz)")
    ax[1].set_title(f"Pairwise influence (fitted slope for d≥6: {slope:.2f})"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig("results/Lk_vs_bound.png", dpi=140)


if __name__ == "__main__":
    main()
