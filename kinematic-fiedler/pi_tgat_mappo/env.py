"""
UAVSwarmEnv: cooperative multi-agent environment approximating the
manuscript's Dec-POMDP (Methods, eq. for S, A, Omega, O, T, R, gamma).

Physics: each drone is a velocity-controlled point mass (the paper's actor
also outputs a 3D target velocity vector v_target, sent to the flight
controller as a MAVLink SET_POSITION_TARGET_LOCAL_NED command -- here we
integrate that same velocity command directly with a first-order lag,
rather than driving an actual ArduPilot SITL instance).

Task: the whole swarm shares a sequence of K waypoints; each agent
independently advances its own progress index through that same sequence
(individual task reward), while a shared, team-level spectral connectivity
penalty (Fiedler value) is added to every agent's reward every step,
mirroring the paper's global reward function R(s,a) = R_task + omega *
R_conn(lambda_2(L)).

Communication / graph construction: a link (i, j) exists if the pair is
within a soft RF range governed by a logistic drop-probability model --
Table 1 explicitly describes kappa as "controls **logistic** drop
probability steepness beyond R_comm", so packet delivery per step is
Bernoulli(1 - P_drop(d_ij)), not a hard geometric cutoff.

Every timestep, each receiving agent i gets ONE incoming edge per other
agent j: either a REAL edge (link up: carries j's true relative position /
velocity) or, when use_kinematic_prior is on, a SYNTHETIC edge (link down:
carries the raw Reynolds alignment/cohesion/separation components computed
from i's cached neighborhood). Combining those three components into an
actual synthetic velocity is left to the ACTOR NETWORK, which owns the
learnable w_align/w_cohesion/w_separation weights (paper: "we formulate
w1, w2, w3 as learnable scalar parameters embedded directly in the PI-TGAT
actor network... jointly optimized via backpropagation") -- the environment
only supplies the raw physical ingredients, never the fixed weighting.
With use_kinematic_prior off, a down link simply produces no edge at all --
the memoryless-GAT baseline the paper compares against.
"""

from __future__ import annotations
import numpy as np

from .config import Config

# Edge feature layout (dim = 17), identical for every edge; unused fields are zero:
#   [0:3]   true relative position (neighbor - self), normalized by box diagonal (real edges only)
#   [3:6]   true relative velocity (neighbor - self), normalized by v_max        (real edges only)
#   [6:9]   Reynolds "alignment" component: mean velocity of i's known neighborhood, normalized (synthetic only)
#   [9:12]  Reynolds "cohesion" component: steer-to-centroid vector, normalized                 (synthetic only)
#   [12:15] Reynolds "separation" component: inverse-distance repulsion vector, normalized       (synthetic only)
#   [15]    time since this info was actually received, normalized (real edges -> 0)
#   [16]    is_synthetic flag (0 = real packet, 1 = kinematic-prior projection)
EDGE_FEAT_DIM = 17
NODE_FEAT_DIM = 7  # [rel_pos_to_waypoint(3), velocity(3), battery(1)]
DT_NORM_SCALE = 5.0  # seconds; keeps the dt_since feature in a sane range


def logistic_drop_prob(dist: np.ndarray, r_comm: float, kappa: float) -> np.ndarray:
    """P_drop(d) = sigma(kappa * (d - r_comm)); ~0 well inside range, ~1 well beyond."""
    return 1.0 / (1.0 + np.exp(-kappa * (dist - r_comm)))


class UAVSwarmEnv:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.node_feat_dim = NODE_FEAT_DIM
        self.edge_feat_dim = EDGE_FEAT_DIM

        self.n = None
        self.pos = None
        self.vel = None
        self.battery = None
        self.waypoints = None          # (K, 3) shared mission path
        self.wp_idx = None             # (n,) each agent's current target waypoint index
        self.t = 0
        # Per-(observer i, subject j) cache of the last packet i actually received from j.
        self.last_seen_pos = None      # (n, n, 3)
        self.last_seen_vel = None      # (n, n, 3)
        self.last_seen_t = None        # (n, n) step index of last successful packet

    # ------------------------------------------------------------------
    def reset(self, n_agents: int | None = None):
        cfg = self.cfg
        self.n = n_agents or int(self.rng.integers(cfg.n_min, cfg.n_max + 1))
        box = np.asarray(cfg.box_size)
        self.pos = self.rng.uniform(0, 1, size=(self.n, 3)) * box
        self.vel = np.zeros((self.n, 3))
        self.battery = np.ones(self.n)
        self.waypoints = self.rng.uniform(0.1, 0.9, size=(cfg.n_waypoints, 3)) * box
        self.wp_idx = np.zeros(self.n, dtype=np.int64)
        self.t = 0

        # Assume the swarm is fully aware of everyone's initial state at t=0
        # (a one-time pre-mission sync), so the kinematic prior has something
        # to work from even before the first real packet exchange.
        self.last_seen_pos = np.repeat(self.pos[None, :, :], self.n, axis=0)
        self.last_seen_vel = np.zeros((self.n, self.n, 3))
        self.last_seen_t = np.zeros((self.n, self.n), dtype=np.int64)
        return self._build_graph()

    # ------------------------------------------------------------------
    def _pairwise_dist(self):
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def _node_features(self):
        cfg = self.cfg
        box_norm = np.linalg.norm(cfg.box_size)
        targets = self.waypoints[np.clip(self.wp_idx, 0, cfg.n_waypoints - 1)]
        rel_to_goal = (targets - self.pos) / (box_norm + 1e-8)
        vel_norm = self.vel / (cfg.v_max + 1e-8)
        return np.concatenate([rel_to_goal, vel_norm, self.battery[:, None]], axis=1).astype(np.float32)

    def _build_graph(self):
        """Returns node features plus ONE unified edge list: a real edge for
        every currently-up link, a synthetic (kinematic-prior) edge for every
        currently-down link (if enabled)."""
        cfg = self.cfg
        box_norm = np.linalg.norm(cfg.box_size)
        v_norm = cfg.v_max + 1e-8
        dist = self._pairwise_dist()
        p_drop = logistic_drop_prob(dist, cfg.r_comm, cfg.kappa)
        np.fill_diagonal(p_drop, 1.0)
        link_up = self.rng.uniform(size=p_drop.shape) >= p_drop
        np.fill_diagonal(link_up, False)

        # Evaluation-only stress test (see config.node_dropout_rate): on top of
        # the organic, distance-based RF link dropout above, independently
        # silence a fraction of AGENTS entirely this step -- every link
        # originating from a silenced agent fails, regardless of distance.
        # This is the paper's "simultaneous node dropout" evaluation
        # condition (Table 1: 0% / 15% / 30%), distinct from per-link RF drop.
        if cfg.node_dropout_rate > 0.0:
            silenced = self.rng.uniform(size=self.n) < cfg.node_dropout_rate
            link_up[:, silenced] = False

        # Update the temporal cache for links that ARE up this step (i receives from j).
        ii, jj = np.where(link_up)
        self.last_seen_pos[ii, jj] = self.pos[jj]
        self.last_seen_vel[ii, jj] = self.vel[jj]
        self.last_seen_t[ii, jj] = self.t

        node_feats = self._node_features()
        edges_src, edges_dst, edges_attr = [], [], []

        # Real edges: i <- j, i receives j's true current relative state.
        i_idx, j_idx = np.where(link_up)
        if i_idx.size:
            rel_pos = (self.pos[j_idx] - self.pos[i_idx]) / (box_norm + 1e-8)
            rel_vel = (self.vel[j_idx] - self.vel[i_idx]) / v_norm
            zeros9 = np.zeros((len(i_idx), 9))
            dt_since = np.zeros((len(i_idx), 1))
            is_synth = np.zeros((len(i_idx), 1))
            edges_src.append(j_idx); edges_dst.append(i_idx)
            edges_attr.append(np.concatenate([rel_pos, rel_vel, zeros9, dt_since, is_synth], axis=1))

        # Synthetic edges: i <- j, raw Reynolds components from i's cached neighborhood.
        down_i, down_j = np.where((~link_up) & ~np.eye(self.n, dtype=bool))
        if down_i.size and cfg.use_kinematic_prior:
            align, cohesion, separation = self._reynolds_components(down_i, down_j)
            zeros6 = np.zeros((len(down_i), 6))
            dt_since = ((self.t - self.last_seen_t[down_i, down_j]) * cfg.dt / DT_NORM_SCALE)[:, None]
            is_synth = np.ones((len(down_i), 1))
            edges_src.append(down_j); edges_dst.append(down_i)
            edges_attr.append(np.concatenate(
                [zeros6, align / v_norm, cohesion / (box_norm + 1e-8), separation / (box_norm + 1e-8),
                 dt_since, is_synth], axis=1))

        if edges_src:
            edge_index = np.stack(
                [np.concatenate(edges_src), np.concatenate(edges_dst)], axis=0
            ).astype(np.int64)
            edge_attr = np.concatenate(edges_attr, axis=0).astype(np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

        return dict(
            node_feats=node_feats,     # (n, 7)
            edge_index=edge_index,      # (2, E) src -> dst, "dst receives from src"
            edge_attr=edge_attr,        # (E, 17)
            positions=self.pos.copy(),
        )

    def _reynolds_components(self, down_i, down_j):
        """For each (i, j) with a dropped link, compute the three RAW Reynolds
        ingredients (alignment, cohesion, separation) from i's last-known
        local neighborhood -- unweighted; the network applies learnable
        weights to combine them. Returns three (len(down_i), 3) arrays."""
        n = self.n
        align = np.zeros((len(down_i), 3))
        cohesion = np.zeros((len(down_i), 3))
        separation = np.zeros((len(down_i), 3))
        for i in np.unique(down_i):
            mask = down_i == i
            js = down_j[mask]
            known = np.setdiff1d(np.arange(n), [i])
            neigh_pos = self.last_seen_pos[i, known]
            neigh_vel = self.last_seen_vel[i, known]
            a = neigh_vel.mean(axis=0)                                    # alignment: mean neighbor velocity
            c = neigh_pos.mean(axis=0) - self.last_seen_pos[i, js]        # cohesion: steer toward centroid
            diffs = self.last_seen_pos[i, js][:, None, :] - neigh_pos[None, :, :]
            d = np.linalg.norm(diffs, axis=-1) + 1e-3
            s = (diffs / d[..., None] ** 2).mean(axis=1)                  # separation: inverse-distance repulsion
            align[mask] = a
            cohesion[mask] = c
            separation[mask] = s
        return align, cohesion, separation

    # ------------------------------------------------------------------
    def fiedler_value(self) -> float:
        """Algebraic connectivity of the SOFT (reliability-weighted) Laplacian,
        i.e. w_ij = 1 - P_drop(d_ij) rather than a hard 0/1 adjacency. This
        keeps lambda_2 continuous in agent position (per the manuscript's
        "continuous differentiability through the spectral theorem"), and
        avoids the piecewise-constant behaviour a hard in/out-of-range
        adjacency would give -- see kinematic-fiedler/fiedler_sim.py, panel C,
        which demonstrates exactly that failure mode for a binary adjacency."""
        dist = self._pairwise_dist()
        w = 1.0 - logistic_drop_prob(dist, self.cfg.r_comm, self.cfg.kappa)
        np.fill_diagonal(w, 0.0)
        L = np.diag(w.sum(axis=1)) - w
        eigvals = np.linalg.eigvalsh(L)
        return float(max(eigvals[1], 0.0)) if len(eigvals) > 1 else 0.0

    # ------------------------------------------------------------------
    def step(self, actions):
        """actions: (n, 3) float array in [-1, 1]^3, the manuscript's
        normalized v_target action head (scaled here by cfg.v_max)."""
        cfg = self.cfg
        target_vel = np.clip(actions, -1.0, 1.0) * cfg.v_max
        self.vel = 0.7 * self.vel + 0.3 * target_vel   # first-order lag (avoids teleport-like jumps)
        self.pos = np.clip(self.pos + self.vel * cfg.dt, 0, np.array(cfg.box_size))
        self.battery = np.clip(self.battery - 1e-4, 0, 1)
        self.t += 1

        dist_to_target = np.linalg.norm(
            self.waypoints[np.clip(self.wp_idx, 0, cfg.n_waypoints - 1)] - self.pos, axis=1
        )
        box_norm = np.linalg.norm(cfg.box_size)
        task_reward = -dist_to_target / box_norm

        reached = (dist_to_target < cfg.waypoint_radius) & (self.wp_idx < cfg.n_waypoints - 1)
        task_reward = task_reward + reached.astype(np.float64) * 1.0
        self.wp_idx = np.minimum(self.wp_idx + reached.astype(np.int64), cfg.n_waypoints - 1)
        all_done = self.wp_idx >= cfg.n_waypoints - 1
        finished_now = all_done & (dist_to_target < cfg.waypoint_radius)

        lam2 = self.fiedler_value()
        r_conn = -max(0.0, np.exp(cfg.lambda_crit - lam2) - 1.0)
        reward = task_reward + cfg.omega * r_conn   # shared team term added to every agent

        done = self.t >= cfg.max_steps or bool(np.all(finished_now))
        info = dict(
            fiedler_value=lam2,
            connected=lam2 > 0.0,
            mission_progress=float(np.mean(self.wp_idx) / (cfg.n_waypoints - 1)),
            mission_complete=bool(np.all(finished_now)),
        )
        return self._build_graph(), reward, done, info
