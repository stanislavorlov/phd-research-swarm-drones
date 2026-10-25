"""
make_method_figs.py -- the three schematics for the Methods section.

    python make_method_figs.py                 # writes PDF + SVG + PNG into figs/
    python make_method_figs.py --out figs --dpi 400

Fig 1  the pipeline: planner -> BC/DAgger distillation -> MAPPO fine-tune, with the sample budget underneath.
Fig 2  the observation and why it binds: what the agent sees (a) vs what determines its reward (b).
Fig 3  the set-based actor/critic: the only difference between the two is the attention mask.

Vector output (PDF/SVG) for the manuscript; PNG only for previewing. Palette is colour-blind safe and survives
greyscale printing (the three accents differ in lightness, not only in hue).
NOTE: written without being executed by its author -- check the rendering before using it in the paper.
"""
from __future__ import annotations
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style
INK   = "#1a1a1a"      # text / outlines
MUTE  = "#9a9a9a"      # out-of-range entities, de-emphasised rules
BLUE  = "#21456e"      # dark   -- the agent / offline stage
TEAL  = "#3f8f8a"      # mid    -- users / distillation stage
ORANG = "#c76a1e"      # light  -- interference / online stage
PALE  = "#eef1f4"      # box fill

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8.0, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})



_BOXES = []          # (x, y, w, h, Text) registered by box(); checked by fit_check()


def fit_check(fig, ax, name):
    """Measure every box label against its box and report overflow instead of leaving it to the eye."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for (x, y, w, h, tx) in _BOXES:
        bb = tx.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        if bb.width > w - 0.004 or bb.height > h - 0.004:
            bad.append(f"      overflow in {name!r}: {tx.get_text()[:38]!r} "
                       f"needs {bb.width:.3f}x{bb.height:.3f}, box is {w:.3f}x{h:.3f}")
    _BOXES.clear()
    for b in bad:
        print(b)
    return not bad


def save(fig, out, name, dpi):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}/{name}.{{pdf,svg,png}}")


def box(ax, x, y, w, h, text, fc=PALE, ec=INK, lw=0.8, fs=8.0, weight="normal", tc=INK, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    tx = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                 color=tc, weight=weight, zorder=3, linespacing=1.35)
    _BOXES.append((x, y, w, h, tx))
    return tx


def arrow(ax, p0, p1, color=INK, lw=0.9, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, lw=lw, ls=ls,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5, zorder=4))


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- Fig 1: pipeline
def fig_pipeline(out, dpi):
    """Three stages left to right. Explanatory detail belongs in the caption, not in the boxes."""
    fig, ax = blank(7.1, 2.45)
    cols = [(0.000, 0.275, BLUE, "Stage 1  offline dataset"),
            (0.3625, 0.275, TEAL, "Stage 2  distillation"),
            (0.7250, 0.275, ORANG, "Stage 3  online fine-tuning")]
    txt = [("receding-horizon planner\n(block coordinate descent)",
            "600 scenarios, open-loop\nreplayed through the env"),
           ("behaviour cloning\n(round 0)",
            "DAgger rounds 1-8\nplanner re-solved at\nevery visited state"),
           ("MAPPO fine-tuning\nmasked vs unmasked critic",
            "coverage constraint\n(training reward only)")]
    yt, ht, yb, hb = 0.655, 0.195, 0.355, 0.215
    for (x, w, c, head), (a, b) in zip(cols, txt):
        ax.text(x, 0.945, head, fontsize=8.0, weight="bold", color=c)
        ax.add_line(Line2D([x, x + w], [0.915, 0.915], color=c, lw=1.0))
        box(ax, x, yt, w, ht, a, fc="#f7f9fb", ec=c, fs=6.9)
        box(ax, x, yb, w, hb, b, fs=6.9)
        arrow(ax, (x + w / 2, yt), (x + w / 2, yb + hb), color=c)
    for x0, x1, lab in ((0.275, 0.3625, "dataset"), (0.6375, 0.7250, "k-masked\nstudent")):
        arrow(ax, (x0, yb + hb / 2), (x1, yb + hb / 2))
        ax.text((x0 + x1) / 2, yb + hb / 2 + 0.085, lab, fontsize=6.3, ha="center", va="bottom")

    y = 0.145
    ax.text(0.0, y + 0.115, "sample budget (team steps)", fontsize=7.4, weight="bold")
    scale, bh = 0.52 / 1_132_800, 0.045
    x = 0.0
    for v, c in ((52_960, BLUE), (288_000, ORANG)):
        ax.add_patch(Rectangle((x, y), scale * v, bh, fc=c, ec="none", zorder=2)); x += scale * v
    ax.text(x + 0.012, y + bh / 2, "distilled pipeline  341k  (offline 53k + online 288k)",
            fontsize=6.9, va="center")
    ax.add_patch(Rectangle((0.0, y - 0.085), 0.52, bh, fc=MUTE, ec="none", zorder=2))
    ax.text(0.532, y - 0.085 + bh / 2, "from scratch, to the same level  1.13M  (3.3x)",
            fontsize=6.9, va="center")
    ok = fit_check(fig, ax, "pipeline")
    save(fig, out, "fig_methods_pipeline", dpi)
    return ok


# ---------------------------------------------------------------- Fig 2: observation vs reward
def _scene(seed=3, N=8, U=26, W=20.0):
    rng = np.random.default_rng(seed)
    cen = rng.uniform(0.18 * W, 0.82 * W, (4, 2))
    users = np.clip(cen[rng.integers(0, 4, U)] + rng.normal(0, 1.9, (U, 2)), 0, W)
    uav = rng.uniform(0.1 * W, 0.9 * W, (N, 2))
    uav[0] = [0.46 * W, 0.52 * W]                      # focal agent
    ch = rng.integers(0, 3, N); ch[0] = 0
    ch[np.argmax(np.linalg.norm(uav - uav[0], axis=1))] = 0   # a far peer shares the focal channel
    return users, uav, ch, W


def fig_observation(out, dpi):
    users, uav, ch, W = _scene()
    k = 6.0
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    foc = uav[0]
    served = np.argsort(np.linalg.norm(users - foc, axis=1))[:5]

    for ax in axes:
        ax.set_xlim(-0.6, W + 0.6); ax.set_ylim(-0.6, W + 0.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), W, W, fc="none", ec=MUTE, lw=0.6))

    # (a) what the agent sees
    ax = axes[0]
    ax.add_patch(Circle(foc, k, fc="#f4f7fa", ec=BLUE, lw=0.9, ls=(0, (3, 2)), zorder=1))
    du = np.linalg.norm(users - foc, axis=1); da = np.linalg.norm(uav - foc, axis=1)
    ax.scatter(*users[du > k].T, s=11, marker="o", fc="none", ec=MUTE, lw=0.7, zorder=3)
    ax.scatter(*users[du <= k].T, s=13, marker="o", c=TEAL, zorder=3)
    far = (da > k); far[0] = False
    ax.scatter(*uav[far].T, s=30, marker="^", fc="none", ec=MUTE, lw=0.8, zorder=3)
    near = (da <= k); near[0] = False
    ax.scatter(*uav[near].T, s=32, marker="^", c=BLUE, zorder=3)
    ax.scatter(*foc, s=72, marker="^", c=BLUE, ec="white", lw=0.9, zorder=5)
    ax.text(foc[0] - 0.6, foc[1] - 1.9, r"agent $i$", fontsize=7.4, color=BLUE, weight="bold", ha="center")
    ax.annotate("", xy=foc, xytext=foc + np.array([k * 0.72, k * 0.69]),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(foc[0] + k * 0.30, foc[1] + k * 0.47, r"$k$", fontsize=8.2, color=BLUE)
    ax.set_title("(a) what the agent observes", fontsize=8.6, pad=5)

    # (b) what determines the reward
    ax = axes[1]
    ax.add_patch(Circle(foc, k, fc="none", ec=MUTE, lw=0.7, ls=(0, (3, 2)), zorder=1))
    ax.scatter(*users.T, s=11, marker="o", c=MUTE, zorder=3)
    ax.scatter(*users[served].T, s=15, marker="o", c=TEAL, zorder=4)
    co = np.where(ch == ch[0])[0]; co = co[co != 0]
    other = [j for j in range(len(uav)) if j != 0 and j not in co]
    ax.scatter(*uav[other].T, s=28, marker="^", fc="none", ec=MUTE, lw=0.8, zorder=3)
    ax.scatter(*uav[co].T, s=34, marker="^", c=ORANG, zorder=4)
    ax.scatter(*foc, s=72, marker="^", c=BLUE, ec="white", lw=0.9, zorder=5)
    for j in co[:2]:
        for u in served[:2]:
            ax.annotate("", xy=users[u], xytext=uav[j],
                        arrowprops=dict(arrowstyle="-|>", color=ORANG, lw=0.65, alpha=0.75,
                                        shrinkA=3, shrinkB=2, mutation_scale=7))
    for u in served:
        ax.plot([foc[0], users[u][0]], [foc[1], users[u][1]], color=TEAL, lw=0.7, zorder=2)
    jf = co[np.argmax(np.linalg.norm(uav[co] - foc, axis=1))]
    ax.annotate("co-channel peer\noutside the radius", xy=uav[jf], xytext=(uav[jf][0] - 2.4, uav[jf][1] + 2.6),
                fontsize=6.9, color=ORANG, ha="center",
                arrowprops=dict(arrowstyle="-", color=ORANG, lw=0.6))
    ax.set_title("(b) what determines its reward", fontsize=8.6, pad=5)

    handles = [Line2D([], [], marker="^", ls="none", c=BLUE, ms=6, label="agent / visible peer"),
               Line2D([], [], marker="^", ls="none", mfc="none", mec=MUTE, ms=6, label="peer outside radius"),
               Line2D([], [], marker="^", ls="none", c=ORANG, ms=6, label="co-channel peer"),
               Line2D([], [], marker="o", ls="none", c=TEAL, ms=5, label="user (served / visible)"),
               Line2D([], [], ls="-", c=ORANG, lw=0.9, label="interference")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, -0.045), handletextpad=0.5, columnspacing=1.5)
    fig.text(0.5, -0.115,
             r"Interference sums over $\it{all}$ co-channel peers with no distance cut-off ($\alpha=2.5$), "
             r"so a peer the agent cannot see still sets its users' SINR.",
             ha="center", fontsize=7.3, color=INK)
    save(fig, out, "fig_methods_observation", dpi)
    return True


# ---------------------------------------------------------------- Fig 3: architecture
def fig_architecture(out, dpi):
    """Encoder once, two consumers. Everything explanatory lives in the caption."""
    fig, ax = blank(7.1, 2.95)
    ys = [0.735, 0.475, 0.215]

    xi, wi = 0.0, 0.225
    ax.text(xi, 0.925, "radius-limited observation", fontsize=8.0, weight="bold")
    for y, lab in zip(ys, [r"self  $s_i$" "\n" r"position, channel, $t/T$, $\beta_i$",
                           r"users  $\{f_{u,i}\}$" "\nrel. position, velocity",
                           r"peers  $\{h_{j,i}\}$" "\nrel. position, channel"]):
        box(ax, xi, y, wi, 0.155, lab, fs=6.9)

    xm, wm = 0.275, 0.095
    for y, lab in zip(ys, [r"MLP$_s$", r"MLP$_u$", r"MLP$_a$"]):
        box(ax, xm, y + 0.0375, wm, 0.08, lab, fs=7.2)
        arrow(ax, (xi + wi, y + 0.0775), (xm, y + 0.0775))

    xa, wa = 0.425, 0.215
    box(ax, xa, 0.595, wa, 0.215, "multi-head attention\nkeys: null, users, peers\nkey mask = visibility",
        fc="#f7f9fb", ec=BLUE, fs=6.9)
    box(ax, xa, 0.275, wa, 0.215, "sum-pool\nfixed normalisers\ncounts survive", fs=6.9)
    arrow(ax, (xm + wm, ys[0] + 0.078), (xa, 0.70), rad=0.05)
    for y, y_att, y_pool in ((ys[1], 0.660, 0.420), (ys[2], 0.620, 0.345)):
        arrow(ax, (xm + wm, y + 0.078), (xa, y_att), rad=0.06)
        arrow(ax, (xm + wm, y + 0.078), (xa, y_pool), rad=-0.05)

    xc, wc = 0.675, 0.085
    box(ax, xc, 0.445, wc, 0.17, r"MLP$_o$" "\n" r"$\to z_i$", fs=7.2)
    arrow(ax, (xa + wa, 0.70), (xc, 0.575), rad=0.07)
    arrow(ax, (xa + wa, 0.38), (xc, 0.485), rad=-0.07)

    xh, wh = 0.800, 0.200
    for y, lab, tc, ec in ((0.735, "MDN position head", INK, INK),
                           (0.580, "channel logits", INK, INK),
                           (0.425, "chunk head (aux)", MUTE, MUTE)):
        box(ax, xh, y, wh, 0.105, lab, fs=6.9, tc=tc, ec=ec)
        arrow(ax, (xc + wc, 0.53), (xh, y + 0.0525), rad=0.06)
    box(ax, xh, 0.185, wh, 0.185, r"critic  $V_\phi$" "\nseparate instance of\nthe same encoder",
        fc="#fdf5ee", ec=ORANG, fs=6.9)
    ax.add_line(Line2D([xh - 0.035, xh - 0.035], [0.185, 0.370], color=ORANG, lw=0.8, ls=(0, (2.2, 1.5))))
    ax.text(xi, 0.075,
            "Masked and unmasked critics differ in the key mask alone: same class, same architecture,\n"
            "same initialisation under the same seed. The critic is a separate instance, not a second head.",
            fontsize=6.9, color=ORANG, va="top")
    ok = fit_check(fig, ax, "architecture")
    save(fig, out, "fig_methods_architecture", dpi)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    ok = [fig_pipeline(a.out, a.dpi), fig_observation(a.out, a.dpi), fig_architecture(a.out, a.dpi)]
    print("  all labels fit" if all(ok) else "  SOME LABELS OVERFLOW -- see above")


if __name__ == "__main__":
    main()
