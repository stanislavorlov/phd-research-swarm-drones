"""make_p2_figs.py -- Figures 2-5 of paper 2 from files already on disk. numpy + matplotlib only; no torch.

    python3 make_p2_figs.py                         # all figures it has data for, into figs/
    python3 make_p2_figs.py --only 3 4              # just some
    python3 make_p2_figs.py --root results2 --adv data/learned_adv_L125_s0.npz

Fig 2 needs data/learned_adv_*.npz from dump_learned_adv.py (one critic re-train). Figs 3-5 read only
results2/: final_eval_*_strong++.json, M_mix/seed*/log.txt, online/ft_*/run_masked_seed*.npz,
<arm>/seed*/{summary.json,log.txt}. Every number drawn is also printed, so the captions can be checked.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette shared with fig1 (paired-counterfactual schematic)
BLUE, ORANGE, AQUA, PURPLE, INK, INK2, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6",
                                                      "#0b0b0b", "#52514e", "#9a9892", "#dcdbd6")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "legend.frameon": False})
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), T95.get(len(x), 1.96) * float(x.std(ddof=1)) / np.sqrt(len(x))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/{name}.png/.pdf")


# ----------------------------------------------------------------------------- Fig 2
def fig2(a):
    f = a.adv or (sorted(glob.glob("data/learned_adv_*.npz")) or [None])[0]
    if not f or not os.path.exists(f):
        print("Fig 2: no learned_adv npz -- run dump_learned_adv.py first; skipped"); return
    d = np.load(f)
    S, T, N = int(d["S"]), int(d["T"]), int(d["N"])
    eA, eB = d["exact_adv_a"], d["exact_adv_b"]                       # [S, T], per state
    L = d["learned_adv"].reshape(T, 2, S, N).mean(-1)                 # pair_rows order: (t, branch, s, agent)
    lA, lB = L[:, 0].T, L[:, 1].T                                     # [S, T]
    eg, lg = (eA - eB).ravel(), (lA - lB).ravel()
    print(f"Fig 2 ({os.path.basename(f)}): exact gap {eg.mean():+.3f} (sd {eg.std():.3f}, weak better at "
          f"{100 * (eg < 0).mean():.1f}%)   learned gap {lg.mean():+.4f} (sd {lg.std():.4f})   "
          f"corr {np.corrcoef(eg, lg)[0, 1]:+.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.42))
    for k, (A, B, ttl) in enumerate(((eA, eB, "(a) exact, from the planner"), (lA, lB, "(b) learned IQL critic"))):
        lo, hi = np.percentile(np.r_[A.ravel(), B.ravel()], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for X, c, lab in ((A, BLUE, "branch A (strong)"), (B, ORANGE, "branch B (weak)")):
            ax[k].hist(X.ravel(), bins=bins, color=c, alpha=0.45, lw=0, label=lab)
            ax[k].axvline(X.mean(), color=c, lw=1.2)
        g = A.mean() - B.mean()
        ytop = ax[k].get_ylim()[1] * 1.22
        ax[k].set_ylim(0, ytop)
        ax[k].annotate("", xy=(A.mean(), 0.86 * ytop), xytext=(B.mean(), 0.86 * ytop),
                       arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=0, shrinkB=0))
        ax[k].text((A.mean() + B.mean()) / 2, 0.895 * ytop, f"gap {g:+.3f}", ha="center", va="bottom", fontsize=8,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax[k].set_title(ttl, fontsize=8.5, loc="left"); ax[k].set_xlabel("advantage, per state"); ax[k].set_yticks([])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.36, -0.2))
    m = np.percentile(np.abs(eg), 99.5)
    ax[2].scatter(eg, lg, s=2, color=INK2, alpha=0.25, lw=0)
    ax[2].axhline(0, color=GRID, lw=0.8, zorder=0); ax[2].axvline(0, color=GRID, lw=0.8, zorder=0)
    ax[2].plot([-m, m], [-m, m], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax[2].set_xlim(-m, m); ax[2].set_ylim(-m, m)
    ax[2].set_xlabel(r"exact $Q_A - Q_B$"); ax[2].set_ylabel(r"learned $Q_A - Q_B$")
    ax[2].set_title("(c) branch gap, state by state", fontsize=8.5, loc="left")
    ax[2].text(0.03, 0.95, f"r = {np.corrcoef(eg, lg)[0, 1]:+.2f}", transform=ax[2].transAxes, va="top", fontsize=8)
    save(fig, a.out, "fig_advantages")


# ----------------------------------------------------------------------------- Fig 3
OFFLINE = [  # (label, file tag, key)  -- order is cosmetic; the plot sorts by mean
    ("P_hard: keep better branch", "offline5_test", "P_hard"),
    ("Q_strong: BC, clean data only", "quality_test", "Q_strong"),
    (r"P05: AWR, exact adv., $\beta$=0.5", "offline5_test", "P05_pairs"),
    ("B_mix: BC, mixed data", "offline5_test", "B_mix"),
    ("P_rand: keep random branch", "offline5_test", "P_rand"),
    ("P0: BC, paired data", "offline5_test", "P0_pairs"),
    (r"L125: IQL, learned adv., $\beta$=1.25", "offline5_test", "L125_pairs"),
    ("M_mix: MADT, mixed data", None, "M_mix"),
]
FINAL = re.compile(r"\[final\].*return\s+([\d.]+)\s+norm\s+([\d.]+).*\(stay\s+([\d.]+),\s+den\s+([\d.]+)\)")


def load_final(root, tag):
    p = os.path.join(root, "final", f"final_eval_{tag}_strong++.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig3(a):
    rows, stay, den = [], None, None
    for lab, tag, key in OFFLINE:
        if tag is None:                                                 # MADT: last [final] line of each log
            r = []
            for lg in sorted(glob.glob(os.path.join(a.root, key, "seed*", "log.txt"))):
                hits = [FINAL.search(l) for l in open(lg, errors="replace")]
                hits = [h for h in hits if h]
                if hits:
                    r.append(float(hits[-1].group(1)))
                    st_, dn_ = float(hits[-1].group(3)), float(hits[-1].group(4))
            if not r:
                print(f"Fig 3: no [final] lines for {key}; skipped"); continue
            ret = np.array(r)
        else:
            d = load_final(a.root, tag)
            if d is None or key not in d["results"]:
                print(f"Fig 3: {key} not in final_eval_{tag}; skipped"); continue
            ret = np.array(d["results"][key]["returns"], float)
            st_, dn_ = d["stay"], d["expert"] - d["stay"]
        if stay is None:
            stay, den = st_, dn_
        elif abs(st_ - stay) > 0.05 or abs(dn_ - den) > 0.05:
            raise SystemExit(f"{key}: reference stay/den {st_:.2f}/{dn_:.2f} differs from {stay:.2f}/{den:.2f}")
        norm = (ret - stay) / den                                       # per-student ratio of means
        rows.append((lab, key, ret, norm))
    qd = load_final(a.root, "quality_test")
    qd_norm = None
    if qd and "QD_strong" in qd["results"]:
        qd_norm = (np.mean(qd["results"]["QD_strong"]["returns"]) - stay) / den
    rows.sort(key=lambda r: r[3].mean())
    print(f"Fig 3 (test, n=1024 scenarios, stay {stay:.2f}, den {den:.2f}; interval = t 95% CI over students):")
    for lab, key, ret, norm in rows[::-1]:
        m, h = tci(norm); mr, hr = tci(ret)
        print(f"  {key:<11} n={len(ret)}  return {mr:6.2f} +/- {hr:4.2f}   norm {m:.3f} +/- {h:.3f}")
    if qd_norm is not None:
        print(f"  {'QD_strong':<11} norm {qd_norm:.3f}  (reference line)")
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(rows) + 0.8))
    for y, (lab, key, ret, norm) in enumerate(rows):
        m, h = tci(norm)
        c = BLUE if key in ("P_hard", "P05_pairs") else (PURPLE if key in ("L125_pairs", "M_mix") else INK2)
        jit = (np.arange(len(norm)) - (len(norm) - 1) / 2) * 0.06
        ax.scatter(norm, y + jit, s=9, color=c, alpha=0.45, lw=0, zorder=3)
        ax.errorbar(m, y, xerr=h, fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=1.4, capsize=2.5, zorder=4)
        ax.text(m + h + 0.008, y, f"{m:.3f}", va="center", fontsize=7.5, color=c)
    if qd_norm is not None:
        ax.axvline(qd_norm, color=AQUA, lw=1.1, ls=(0, (4, 2)))
        ax.text(qd_norm - 0.006, len(rows) - 0.55, f"DAgger, strong\nlabels ({qd_norm:.3f})",
                color=AQUA, fontsize=7.5, ha="right", va="top")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("normalised test return  (stay = 0, strong++ planner = 1)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, a.out, "fig_offline_test")


# ----------------------------------------------------------------------------- Fig 4
def fig4(a):
    arms = [("ft_hard", "from P_hard (filtered)", BLUE), ("ft_bc", "from Q_strong (clean-data BC)", AQUA),
            ("ft_scratch", "from random init", ORANGE)]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    print("Fig 4 (monitor, 128 scenarios; mean over scenarios at each evaluation):")
    any_ = False
    for arm, lab, c in arms:
        fs = sorted(glob.glob(os.path.join(a.root, "online", arm, "run_masked_seed*.npz")))
        if not fs:
            print(f"  {arm}: no npz; skipped"); continue
        curves = []
        for f in fs:
            d = np.load(f)
            it, y = d["iters"], d["returns"].mean(1)
            curves.append((it, y))
            ax.plot(it, y, color=c, lw=0.7, alpha=0.45)
        it0 = curves[0][0]
        if all(len(i) == len(it0) and np.all(i == it0) for i, _ in curves):
            Y = np.stack([y for _, y in curves])
            ax.plot(it0, Y.mean(0), color=c, lw=2.0, label=f"{lab}, {len(fs)} seeds")
            print(f"  {arm:<10} it {int(it0[0])}: {Y[:, 0].mean():6.2f}   it {int(it0[-1])}: {Y[:, -1].mean():6.2f}  "
                  f"(per seed {np.round(Y[:, -1], 2)})")
        any_ = True
    if not any_:
        plt.close(fig); return
    ax.axvspan(0, a.warmup, color=GRID, alpha=0.45, lw=0)
    ax.text(a.warmup / 2, 0.50, "critic warm-up\n(actor frozen)", transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=7.5, color=INK2)
    qd = [json.load(open(p))["monitor_return_best"] for p in sorted(glob.glob(os.path.join(a.root, "QD_strong", "seed*", "summary.json")))]
    if qd:
        ax.axhline(np.mean(qd), color=INK2, lw=0.9, ls=(0, (4, 2)))
        ax.text(ax.get_xlim()[1], np.mean(qd) + 1.0, f"DAgger, strong labels ({np.mean(qd):.2f})",
                ha="right", va="bottom", fontsize=7.5, color=INK2)
        print(f"  QD_strong monitor reference {np.mean(qd):.2f} over {len(qd)} students")
    ax.set_xlabel("PPO iteration  (640 team steps each)"); ax.set_ylabel("monitor return")
    ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.67))
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, a.out, "fig_online_curves")


# ----------------------------------------------------------------------------- Fig 5
NEFF = re.compile(r"weights\[(\w+)\].*?p99\s+([\d.]+)\s+max\s+([\d.]+)\s+n_eff\s+(\d+)/(\d+)")
OLD = re.compile(r"AWR weights:.*n_eff\s+(\d+)/(\d+)")
FIG5 = [  # (arm, label, source)
    ("P0_pairs", r"$\beta$=0 (BC)", "solver"), ("P05_pairs", r"$\beta$=0.5", "solver"),
    ("P1_pairs", r"$\beta$=1", "solver"), ("P3_pairs", r"$\beta$=3", "solver"),
    ("P_hard", "hard", "solver"), ("P_rand", "rand", "solver"),
    ("L05_pairs", r"$\beta$=0.5", "learned"), ("L075_pairs", r"$\beta$=0.75", "learned"),
    ("L125_pairs", r"$\beta$=1.25", "learned"),
]


def read_arm(root, arm, source):
    """Per-seed (monitor return, n_eff fraction, p99, max). A seed counts only if its log carries a header written
    by the current iql.py with the matching weight source -- this drops leftovers from earlier code (the stale
    P1_pairs seeds 1-4). beta = 0 is exempt: every version reduces it to unweighted BC."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, arm, "seed*"))):
        sj, lg = os.path.join(d, "summary.json"), os.path.join(d, "log.txt")
        if not (os.path.exists(sj) and os.path.exists(lg)):
            continue
        txt = open(lg, errors="replace").read()
        ok = f"weights={source}/" in txt or arm == "P0_pairs"
        m = list(NEFF.finditer(txt))
        if m:
            neff, p99, mx = int(m[-1].group(4)) / int(m[-1].group(5)), float(m[-1].group(2)), float(m[-1].group(3))
        else:
            o = list(OLD.finditer(txt))
            if not o:
                ok = False; neff = p99 = mx = float("nan")
            else:
                neff, p99, mx = int(o[-1].group(1)) / int(o[-1].group(2)), float("nan"), float("nan")
        if not ok:
            print(f"  (dropped {d}: no '{source}' header from the current code)"); continue
        out.append((json.load(open(sj))["monitor_return_best"], neff, p99, mx))
    return np.array(out) if out else None


def fig5(a):
    print("Fig 5 (monitor return against n_eff):")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    pts = {}
    for arm, lab, src in FIG5:
        r = read_arm(a.root, arm, src)
        if r is None:
            print(f"  {arm}: nothing usable; skipped"); continue
        pts[arm] = r
        m, h = tci(r[:, 0]); x = 100 * r[:, 1].mean()
        c = PURPLE if src == "learned" else (BLUE if arm != "P_rand" else INK2)
        one = len(r) == 1
        ax.scatter(100 * r[:, 1], r[:, 0], s=8, color=c, alpha=0.35, lw=0, zorder=2)
        ax.errorbar(x, m, yerr=None if one else h, fmt="o", ms=5, mfc="white" if one else c, mec=c, color=c,
                    elinewidth=1.2, capsize=2.5, zorder=4)
        dx, dy, ha = {"P_rand": (1.2, -2.4, "left"), "P_hard": (1.2, 1.6, "left"), "P1_pairs": (1.0, -2.6, "left"),
                      "P05_pairs": (-1.2, 1.4, "right")}.get(arm, (1.2, 1.4, "left"))
        ax.text(x + dx, m + dy, lab + (" (1 seed)" if one else ""), fontsize=7.5, color=c, ha=ha)
        clamp = np.nanmean(np.isclose(r[:, 2], r[:, 3])) if np.isfinite(r[:, 2]).any() else 0
        print(f"  {arm:<11} {src:<7} n={len(r)}  n_eff {x:5.1f}%  return {m:6.2f}" + ("" if one else f" +/- {h:.2f}")
              + ("   p99 == max: w_max clamp active" if clamp > 0.5 and arm not in ("P_hard", "P_rand", "P0_pairs") else ""))
    if not pts:
        plt.close(fig); return
    if "P1_pairs" in pts and "P3_pairs" in pts:                         # the clamp pushes n_eff back UP
        p1, p3 = pts["P1_pairs"], pts["P3_pairs"]
        ax.annotate("", xy=(100 * p3[:, 1].mean(), p3[:, 0].mean()), xytext=(100 * p1[:, 1].mean(), p1[:, 0].mean()),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, connectionstyle="arc3,rad=-0.35"))
        ax.text(0.02, 0.985, r"$w_{\max}$ clamp binds at $\beta$=3" "\n" r"(p99 = max), so $n_{\rm eff}$ rises" "\n"
                r"from $\beta$=1 to $\beta$=3", transform=ax.transAxes, fontsize=7.2, color=ORANGE, va="top", ha="left")
    if "P0_pairs" in pts:
        ax.axhline(pts["P0_pairs"][:, 0].mean(), color=GRID, lw=0.9, zorder=0)
    ax.scatter([], [], color=BLUE, s=20, label="exact (planner) advantages")
    ax.scatter([], [], color=PURPLE, s=20, label="learned (IQL) advantages")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_xlabel(r"effective sample size $n_{\rm eff}$ (% of 96,000 rows)"); ax.set_ylabel("monitor return")
    ax.set_xlim(0, 105); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, a.out, "fig_neff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results2")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--adv", default=None, help="npz from dump_learned_adv.py (default: first data/learned_adv_*.npz)")
    ap.add_argument("--warmup", type=int, default=100, help="critic warm-up iterations to shade in Fig 4")
    ap.add_argument("--only", nargs="*", type=int, default=[2, 3, 4, 5])
    a = ap.parse_args()
    for k in a.only:
        {2: fig2, 3: fig3, 4: fig4, 5: fig5}[k](a)


if __name__ == "__main__":
    main()
