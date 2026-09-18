#!/usr/bin/env python3
"""
fiedler_sim.py -- reproduce the diagnostic simulations used in the review of
"Optimization of Swarm Cohesion: Resilient Graph-Based MARL for UAV Swarms
using Kinematic Priors and Spectral Connectivity Rewards".

Self-contained: needs only numpy, scipy, matplotlib.

Two simulations (A, C) and two closed-form checks (B, D):

  A  Distribution of the Fiedler value lambda_2 over random uniform swarm
     deployments, as a function of communication radius R_comm, for the
     manuscript's stated volume (1000 x 1000 x 200 m) and swarm sizes
     N in {30, 100}.  Shows lambda_2 is near-binary and that lambda_crit
     is uninterpretable without a stated R_comm.

  B  The manuscript's connectivity penalty
        R_conn = -max(0, exp(lambda_crit - lambda_2) - 1)
     evaluated at the two lambda_crit values the manuscript states
     (0.2 in Simulation Setup / Table 1, 0.5 in Model Architecture).

  C  lambda_2 along a straight-line translation of ONE UAV through an
     otherwise fixed swarm, under (i) the binary in-range adjacency the
     manuscript specifies and (ii) a smooth sigmoid edge weight.
     Shows lambda_2 is piecewise constant under (i).

  D  For each row of the manuscript's Table 2, the value of tau_safe that
     would be needed to produce the reported MCR from the reported CR
     mean/SD under a Gaussian approximation.

Usage
-----
    python fiedler_sim.py                       # all panels -> fiedler_sim.png
    python fiedler_sim.py --reps 1000           # tighter bands in panel A
    python fiedler_sim.py --print-only          # numbers to stdout, no figure
    python fiedler_sim.py --out myfig.png --dpi 150

Defaults reproduce the figure in the review exactly (seeds 7 and 11).
"""

import argparse
import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------
# Manuscript parameters.  Edit these to test your own configuration.
# --------------------------------------------------------------------------
BOX_A = (1000.0, 1000.0, 200.0)   # stated deployment volume, metres
N_LIST = (30, 100)                # stated swarm sizes, N in {30, 100}
R_SWEEP = np.array([200, 250, 300, 350, 400, 450, 500, 600, 700])  # R_comm, m

LAMBDA_CRIT_TABLE1 = 0.2          # Simulation Setup + Table 1
LAMBDA_CRIT_ARCH   = 0.5          # Model Architecture section

# Panel C: a smaller swarm so individual edge flips are legible.
BOX_C   = (800.0, 800.0, 200.0)
N_C     = 10
RCOMM_C = 450.0
SIGMA_C = 60.0                    # sigmoid width for the smooth-edge variant, m
SWEEP_C = 300.0                   # +/- displacement of the moving UAV, m

# Manuscript Table 2: (node dropout %, arm, CR mean, CR sd, MCR mean, MCR sd)
TABLE2 = [
    (0,  'Memoryless GNN', 98.1, 1.2, 89.5, 2.1),
    (0,  'PI-TGAT',        99.2, 0.8, 94.3, 1.5),
    (15, 'Memoryless GNN', 64.3, 5.4, 61.2, 4.8),
    (15, 'PI-TGAT',        97.5, 2.1, 88.7, 3.2),
    (30, 'Memoryless GNN', 48.7, 8.9, 58.8, 7.1),
    (30, 'PI-TGAT',        95.2, 3.4, 82.4, 4.5),
]


# --------------------------------------------------------------------------
# Core graph-spectral helpers
# --------------------------------------------------------------------------
def fiedler_binary(P, r_comm):
    """lambda_2 of the unweighted graph Laplacian; edge iff pairwise dist < r_comm.

    This is the adjacency the manuscript specifies: "a directed edge e_ij is
    instantiated if agent j remains within sensing or RF range of agent i".
    """
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    A = (D < r_comm).astype(float)
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(1)) - A
    return float(np.linalg.eigvalsh(L)[1])


def fiedler_soft(P, r_comm, sigma):
    """lambda_2 of a Laplacian with smooth (sigmoid) edge weights.

    w_ij = 1 / (1 + exp((d_ij - r_comm) / sigma)).  This is the same functional
    form as the manuscript's packet-drop model P_drop(d_ij), so the weights can
    be read as link reliability.  Unlike fiedler_binary this is differentiable
    in the agent positions.
    """
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    A = 1.0 / (1.0 + np.exp((D - r_comm) / sigma))
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(1)) - A
    return float(np.linalg.eigvalsh(L)[1])


def R_conn(lam2, lam_crit):
    """The manuscript's connectivity penalty, vectorised over lam2."""
    lam2 = np.asarray(lam2, dtype=float)
    return -np.maximum(0.0, np.exp(lam_crit - lam2) - 1.0)


# --------------------------------------------------------------------------
# Panel A -- lambda_2 distribution vs R_comm over random deployments
# --------------------------------------------------------------------------
def sim_lambda2_sweep(reps=300, seed=7, box=BOX_A, n_list=N_LIST, r_sweep=R_SWEEP):
    """For each (N, R_comm), draw `reps` uniform deployments and record lambda_2.

    Returns {N: {R_comm: array of lambda_2}}.  Draws are sequential from one
    Generator, so the default seed reproduces the review figure exactly.
    """
    rng = np.random.default_rng(seed)
    box = np.asarray(box, dtype=float)
    out = {}
    for N in n_list:
        out[N] = {}
        for R in r_sweep:
            vals = np.empty(reps)
            for k in range(reps):
                P = rng.uniform(0.0, 1.0, (N, 3)) * box
                vals[k] = fiedler_binary(P, float(R))
            out[N][int(R)] = np.clip(vals, 0.0, None)
    return out


def summarize_sweep(data, lam_crit=LAMBDA_CRIT_TABLE1):
    """Per-(N, R_comm) median, 10-90% band, P(disconnected), P(in penalty band)."""
    rows = []
    for N, byR in data.items():
        for R, v in byR.items():
            rows.append(dict(
                N=N, R_comm=R,
                median=float(np.median(v)),
                p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)),
                p_disconnected=float(np.mean(v <= 1e-9)),
                p_in_penalty_band=float(np.mean((v > 1e-9) & (v <= lam_crit))),
            ))
    return rows


# --------------------------------------------------------------------------
# Panel C -- lambda_2 along a single-UAV translation
# --------------------------------------------------------------------------
def sim_translation(seed=11, n=N_C, r_comm=RCOMM_C, box=BOX_C, sigma=SIGMA_C,
                    sweep=SWEEP_C, n_steps=1500, min_lam2=0.05, max_tries=30):
    """Translate UAV 0 along +x through an otherwise fixed swarm.

    Rejection-samples an initial configuration with lambda_2 > `min_lam2` so the
    swarm starts connected.  Returns (xs, lam2_binary, lam2_soft, P0).
    """
    rng = np.random.default_rng(seed)
    box = np.asarray(box, dtype=float)
    P0 = None
    for _ in range(max_tries):
        cand = rng.uniform(0.0, 1.0, (n, 3)) * box
        if fiedler_binary(cand, r_comm) > min_lam2:
            P0 = cand
            break
    if P0 is None:
        raise RuntimeError(
            f"no connected start config with lambda_2 > {min_lam2} in "
            f"{max_tries} tries; raise r_comm or shrink box")

    xs = np.linspace(-sweep, sweep, n_steps)
    hard = np.empty(n_steps)
    soft = np.empty(n_steps)
    for i, dx in enumerate(xs):
        P = P0.copy()
        P[0, 0] += dx
        hard[i] = fiedler_binary(P, r_comm)
        soft[i] = fiedler_soft(P, r_comm, sigma)
    return xs, hard, soft, P0


def jump_stats(xs, hard, tol=1e-3):
    """Count and locate the discontinuities in a piecewise-constant lambda_2 trace."""
    d = np.abs(np.diff(hard))
    idx = np.where(d > tol)[0]
    return dict(
        n_jumps=int(idx.size),
        max_jump=float(d.max()) if d.size else 0.0,
        step_m=float(xs[1] - xs[0]),
        jump_positions_m=xs[idx].tolist(),
        lam2_min=float(hard.min()),
        lam2_max=float(hard.max()),
    )


# --------------------------------------------------------------------------
# Panel D -- tau_safe implied by each Table 2 row
# --------------------------------------------------------------------------
def implied_tau_safe(cr_mean, cr_sd, mcr_pct):
    """tau_safe such that P(CR >= tau_safe) = MCR, with CR ~ Normal(mean, sd).

    MCR requires BOTH waypoint success AND CR >= tau_safe, so P(CR >= tau) >= MCR;
    this returns the UPPER BOUND on tau_safe consistent with the reported row.
    """
    return float(cr_mean + norm.ppf(1.0 - mcr_pct / 100.0) * cr_sd)


def table2_consistency(table2=TABLE2):
    rows = []
    for dropout, arm, cr, cr_sd, mcr, mcr_sd in table2:
        rows.append(dict(
            dropout_pct=dropout, arm=arm, CR=cr, CR_sd=cr_sd, MCR=mcr, MCR_sd=mcr_sd,
            tau_safe_upper_bound=implied_tau_safe(cr, cr_sd, mcr),
            mcr_gain_abs_pts=None, mcr_gain_rel_pct=None,
        ))
    # paired relative/absolute MCR gains per dropout level
    for i in range(0, len(rows), 2):
        base, ours = rows[i], rows[i + 1]
        ours['mcr_gain_abs_pts'] = round(ours['MCR'] - base['MCR'], 2)
        ours['mcr_gain_rel_pct'] = round(100.0 * (ours['MCR'] - base['MCR']) / base['MCR'], 2)
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_report(sweep, trans_stats, lam_crit_a=LAMBDA_CRIT_TABLE1,
                 lam_crit_b=LAMBDA_CRIT_ARCH):
    print("=" * 78)
    print("A  lambda_2 over random uniform deployments in "
          f"{BOX_A[0]:.0f} x {BOX_A[1]:.0f} x {BOX_A[2]:.0f} m")
    print("=" * 78)
    print(f"{'N':>5}{'R_comm':>8}{'median':>10}{'p10':>9}{'p90':>9}"
          f"{'P(disc)':>9}{'P(0<L2<=' + str(lam_crit_a) + ')':>16}")
    for r in summarize_sweep(sweep, lam_crit_a):
        print(f"{r['N']:>5}{r['R_comm']:>8}{r['median']:>10.3f}{r['p10']:>9.3f}"
              f"{r['p90']:>9.3f}{r['p_disconnected']:>9.2f}"
              f"{r['p_in_penalty_band']:>16.3f}")
    print("\n  P(disc)          = fraction of deployments with lambda_2 = 0")
    print(f"  P(0<L2<={lam_crit_a})    = fraction inside the ACTIVE penalty band, i.e. the")
    print("                     only region where R_conn produces a gradient.\n")

    print("=" * 78)
    print("B  Connectivity penalty at the two stated lambda_crit values")
    print("=" * 78)
    for lc in (lam_crit_a, lam_crit_b):
        floor = float(R_conn(0.0, lc))
        print(f"  lambda_crit = {lc}:  R_conn floor (lambda_2 -> 0) = {floor:+.4f}"
              f"   dR/dlambda_2 at 0 = {np.exp(lc):.4f}")
    ratio = abs(float(R_conn(0.0, lam_crit_b)) / float(R_conn(0.0, lam_crit_a)))
    print(f"\n  ratio of penalty magnitudes = {ratio:.2f}x")
    print("  -> the two values stated in the manuscript are not interchangeable.\n")

    print("=" * 78)
    print("C  lambda_2 along a translation of one UAV "
          f"(N={N_C}, R_comm={RCOMM_C:.0f} m)")
    print("=" * 78)
    print(f"  binary adjacency : {trans_stats['n_jumps']} discontinuities, "
          f"max single jump {trans_stats['max_jump']:.4f}")
    print(f"  spatial step     : {trans_stats['step_m']:.2f} m")
    print(f"  lambda_2 range   : {trans_stats['lam2_min']:.3f} to {trans_stats['lam2_max']:.3f}")
    print(f"  smooth edges     : max step change {trans_stats['max_step_soft']:.5f}")
    print("  -> lambda_2 is piecewise constant under the specified adjacency;")
    print("     it is differentiable only with smooth edge weights.\n")

    print("=" * 78)
    print("D  tau_safe implied by each row of Table 2")
    print("=" * 78)
    print(f"{'dropout':>9}{'arm':>17}{'CR':>8}{'CR_sd':>7}{'MCR':>7}"
          f"{'tau_safe <=':>13}{'dMCR_abs':>10}{'dMCR_rel':>10}")
    taus = []
    for r in table2_consistency():
        taus.append(r['tau_safe_upper_bound'])
        ga = '' if r['mcr_gain_abs_pts'] is None else f"{r['mcr_gain_abs_pts']:>10.1f}"
        gr = '' if r['mcr_gain_rel_pct'] is None else f"{r['mcr_gain_rel_pct']:>9.1f}%"
        print(f"{r['dropout_pct']:>8}%{r['arm']:>17}{r['CR']:>8.1f}{r['CR_sd']:>7.1f}"
              f"{r['MCR']:>7.1f}{r['tau_safe_upper_bound']:>13.1f}{ga:>10}{gr:>10}")
    print(f"\n  tau_safe must be ONE constant, yet the rows imply "
          f"{min(taus):.0f}-{max(taus):.0f}%  (spread {max(taus)-min(taus):.0f} pts).")
    print("  -> report tau_safe, and split MCR failures into navigation vs connectivity.\n")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(sweep, xs, hard, soft, jstats, out='fiedler_sim.png', dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    BLUE, ORANGE, RED, PURPLE, GREY = '#1f4e9c', '#c05a10', '#c0392b', '#8a2be2', '#8c8c8c'

    fig = plt.figure(figsize=(8.6, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0], hspace=0.55, wspace=0.38)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axD = fig.add_subplot(gs[1, :])

    # A
    Rs = np.array(sorted(next(iter(sweep.values())).keys()))
    for (N, v), c in zip(sorted(sweep.items()), (BLUE, ORANGE)):
        med = np.array([np.median(v[int(R)]) for R in Rs])
        lo = np.array([np.percentile(v[int(R)], 10) for R in Rs])
        hi = np.array([np.percentile(v[int(R)], 90) for R in Rs])
        axA.fill_between(Rs, lo, hi, color=c, alpha=0.18, lw=0)
        axA.plot(Rs, med, '-o', color=c, ms=2.8, lw=1.3, label=f'{N} UAVs')
    axA.axhline(LAMBDA_CRIT_TABLE1, color='#333', ls='--', lw=1.0)
    axA.axhline(LAMBDA_CRIT_ARCH, color=PURPLE, ls=':', lw=1.3)
    axA.set_yscale('symlog', linthresh=0.1)
    axA.set_yticks([0, 0.2, 0.5, 1, 10, 40])
    axA.set_yticklabels(['0', '0.2', '0.5', '1', '10', '40'])
    axA.set_xlabel(r'$R_{comm}$ (m)')
    axA.set_ylabel(r'algebraic connectivity $\lambda_2$')
    axA.text(Rs[-1], LAMBDA_CRIT_TABLE1 * 1.05, '0.2', fontsize=6, color='#333',
             va='bottom', ha='right')
    axA.text(Rs[-1], LAMBDA_CRIT_ARCH * 1.05, '0.5', fontsize=6, color=PURPLE,
             va='bottom', ha='right')
    axA.legend(frameon=False, fontsize=6, loc='lower right',
               bbox_to_anchor=(1.0, 0.03), handlelength=1.1)
    axA.set_title('A  $\\lambda_2$ scale depends entirely on $R_{comm}$,\n'
                  '    which the manuscript never specifies', loc='left')
    axA.margins(x=0.05)

    # B
    g = np.linspace(0, 0.8, 500)
    axB.plot(g, R_conn(g, LAMBDA_CRIT_TABLE1), '--', color='#333', lw=1.6,
             label=rf'$\lambda_{{crit}}={LAMBDA_CRIT_TABLE1}$  (Table 1)')
    axB.plot(g, R_conn(g, LAMBDA_CRIT_ARCH), ':', color=PURPLE, lw=2.0,
             label=rf'$\lambda_{{crit}}={LAMBDA_CRIT_ARCH}$  (§Model Arch.)')
    f1, f2 = float(R_conn(0.0, LAMBDA_CRIT_TABLE1)), float(R_conn(0.0, LAMBDA_CRIT_ARCH))
    axB.axhline(0, color=GREY, lw=0.6)
    axB.plot([0], [f1], 'o', ms=4, color='#333')
    axB.plot([0], [f2], 'o', ms=4, color=PURPLE)
    axB.annotate(f'{f1:.2f}', xy=(0.0, f1), xytext=(0.13, f1 + 0.02), fontsize=6,
                 color='#333', arrowprops=dict(arrowstyle='->', lw=0.6, color='#333'))
    axB.annotate(f'{f2:.2f}', xy=(0.0, f2), xytext=(0.18, f2 - 0.07), fontsize=6,
                 color=PURPLE, arrowprops=dict(arrowstyle='->', lw=0.6, color=PURPLE))
    axB.set_ylim(-0.78, 0.08)
    axB.set_xlabel(r'$\lambda_2$'); axB.set_ylabel(r'$R_{conn}$')
    axB.legend(frameon=False, fontsize=6, loc='center right',
               bbox_to_anchor=(1.03, 0.40), handlelength=1.7)
    axB.set_title(f'B  The two stated $\\lambda_{{crit}}$ differ\n'
                  f'    {abs(f2/f1):.1f}$\\times$ in penalty magnitude', loc='left')

    # C
    d = np.abs(np.diff(hard)); big = np.where(d > 1e-3)[0]
    axC.plot(xs, hard, '-', color=RED, lw=1.2, label='binary edges (as specified)')
    axC.plot(xs, soft, '-', color=BLUE, lw=1.2, label='soft edge weights (differentiable)')
    axC.plot(xs[big], hard[big], '|', color=RED, ms=6, mew=1.0)
    axC.set_xlabel('displacement of one UAV (m)'); axC.set_ylabel(r'$\lambda_2$')
    lo, hi = min(hard.min(), soft.min()), max(hard.max(), soft.max())
    axC.set_ylim(lo - 0.15 * (hi - lo), hi + 0.35 * (hi - lo))
    axC.margins(x=0.03)
    axC.legend(frameon=False, fontsize=6, loc='lower right',
               bbox_to_anchor=(1.02, 0.02), handlelength=1.4)
    axC.text(0.02, 0.97, 'one UAV translated through a fixed swarm;\n'
             'curve magnitudes are not comparable',
             transform=axC.transAxes, fontsize=5.5, color=GREY, va='top')
    axC.set_title('C  Edges are instantiated by a binary in-range /\n'
                  '    packet test, so $\\lambda_2$ is piecewise constant', loc='left')

    # D
    rows = table2_consistency()
    for i, r in enumerate(rows):
        c = GREY if r['arm'] == 'Memoryless GNN' else BLUE
        axD.errorbar(i - 0.15, r['CR'], yerr=r['CR_sd'], fmt='o', ms=4.5,
                     color=c, capsize=2.5, lw=1.2)
        axD.errorbar(i + 0.15, r['MCR'], yerr=r['MCR_sd'], fmt='s', ms=4.5,
                     mfc='white', mec=c, ecolor=c, capsize=2.5, lw=1.2)
        t = r['tau_safe_upper_bound']
        axD.plot(i, t, marker='v', ms=5.5, color=RED, ls='none')
        axD.annotate(f'{t:.0f}', xy=(i, t), xytext=(i + 0.06, t - 1.5),
                     fontsize=5.5, color=RED, ha='left')
    axD.set_xticks(range(len(rows)))
    axD.set_xticklabels([f"{r['arm']}\n{r['dropout_pct']}% dropout" for r in rows],
                        fontsize=6)
    axD.set_ylabel('percent'); axD.set_ylim(35, 106); axD.margins(x=0.05)
    axD.plot([], [], 'o', color='#333', ms=4.5, label='connectivity ratio (CR)')
    axD.plot([], [], 's', color='#333', ms=4.5, mfc='white',
             label='mission completion rate (MCR)')
    axD.plot([], [], 'v', color=RED, ms=5.5, ls='none',
             label=r'$\tau_{safe}$ implied by that row')
    axD.legend(frameon=False, fontsize=6, loc='center left',
               bbox_to_anchor=(0.015, 0.40), handlelength=1.2)
    taus = [r['tau_safe_upper_bound'] for r in rows]
    axD.set_title(r'D  MCR is defined as requiring CR$\,\geq\tau_{safe}$, yet each '
                  rf'reported row implies a different $\tau_{{safe}}$ '
                  rf'({min(taus):.0f}-{max(taus):.0f}%)', loc='left')

    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=300,
                    help='deployments per (N, R_comm) in panel A (default 300)')
    ap.add_argument('--seed-sweep', type=int, default=7, help='seed for panel A')
    ap.add_argument('--seed-trans', type=int, default=11, help='seed for panel C')
    ap.add_argument('--out', default='fiedler_sim.png', help='output figure path')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--print-only', action='store_true',
                    help='print the numbers, skip the figure')
    args = ap.parse_args()

    sweep = sim_lambda2_sweep(reps=args.reps, seed=args.seed_sweep)
    xs, hard, soft, _ = sim_translation(seed=args.seed_trans)
    jstats = jump_stats(xs, hard)
    jstats['max_step_soft'] = float(np.abs(np.diff(soft)).max())

    print_report(sweep, jstats)

    if not args.print_only:
        path = make_figure(sweep, xs, hard, soft, jstats, out=args.out, dpi=args.dpi)
        print(f"figure written to {path}")


if __name__ == '__main__':
    main()
