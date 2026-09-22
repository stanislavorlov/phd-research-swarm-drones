"""
Configuration for the PI-TGAT + MAPPO replication.

Every field below is annotated with (a) the value stated in the manuscript's
Table 1 ("Training parameters for Comprehensive MAPPO Simulation"), and
(b) whether/why it was changed for a runnable implementation. See
README.md for the full parameter-by-parameter mapping table -- this file
is the single source of truth those numbers are pulled from.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---------------------------------------------------------------
    # Physical & RF environment  (Table 1, "Physical & RF Environment")
    # ---------------------------------------------------------------
    n_min: int = 20                     # paper: N in [20, 50]
    n_max: int = 50                     # paper: N in [20, 50]  (sampled per episode)
    box_size: Tuple[float, float, float] = (1000.0, 1000.0, 200.0)   # paper value, unchanged
    r_comm: float = 300.0               # paper: R_comm = 300 m, unchanged
    # Channel decay steepness kappa: the manuscript is internally inconsistent --
    # Table 1 states kappa = 0.02 m^-1, but the Model Architecture section states
    # kappa = 0.01667 m^-1 ("wavelength of 60 m"). We use the Table 1 value since
    # that is the table this experiment was asked to replicate; override via CLI
    # (--kappa) to use the other one.
    kappa: float = 0.02                 # Table 1 value (body text says 0.01667)
    dt: float = 0.1                     # paper: 10 Hz physical step, unchanged
    v_max: float = 15.0                 # NOT in Table 1 (paper never states a physical
                                         # max quadrotor speed). Added so the tanh-squashed
                                         # action head has a concrete m/s scale; 15 m/s is a
                                         # generic small-quadrotor cruise speed. Override with
                                         # --v-max if you have the real number.

    # ---------------------------------------------------------------
    # Task & metric bounds  (Table 1, "Task & Metric Bounds")
    # ---------------------------------------------------------------
    max_steps: int = 150                # paper: T_max = 600 s (6000 steps @ 10 Hz).
                                         # SCALED DOWN to 150 steps (15 s) so a full-episode
                                         # BPTT update through the recurrent GRU is tractable
                                         # on a laptop. Pass --max-steps 6000 to reproduce the
                                         # paper's actual horizon (much slower).
    n_waypoints: int = 6                # paper: K = 6, unchanged
    waypoint_radius: float = 15.0       # paper: r_reach = 15 m, unchanged
    tau_safe: float = 0.85              # paper: tau_safe = 0.85, unchanged
    omega: float = 2.5                  # paper: omega = 2.5 (spectral reward weight), unchanged
    lambda_crit: float = 0.2            # paper: lambda_crit = 0.2 (Table 1 AND Model
                                         # Architecture section agree on this value), unchanged

    # ---------------------------------------------------------------
    # Neural architecture  (Table 1, "Neural Architecture")
    # ---------------------------------------------------------------
    gat_layers: int = 2                 # paper value, unchanged
    gat_heads: int = 4                  # paper value, unchanged
    embed_dim: int = 128                # paper: d_emb = 128, unchanged
    gru_hidden: int = 128               # paper value, unchanged
    decay_rate_init: float = 0.1        # paper: lambda (attention decay) init = 0.1, unchanged
    kinematic_prior_init: Tuple[float, float, float] = (1.0, 1.0, 1.5)
                                         # paper: w_align=1.0, w_cohesion=1.0, w_separation=1.5

    # ---------------------------------------------------------------
    # MAPPO optimization  (Table 1, "MAPPO Optimization")
    # ---------------------------------------------------------------
    n_envs: int = 16                    # paper: 16 parallel SITL workers. We have no SITL and
                                         # no vectorized-env runner, so this is only a *nominal*
                                         # count kept for hyperparameter fidelity; episodes are
                                         # collected sequentially. Effective parallel env count
                                         # used by train.py's rollout collector is
                                         # `episodes_per_iter` (see train.py --episodes-per-iter).
    rollout_length: int = 200           # paper value, unchanged (per-env horizon per update);
                                         # in our sequential collector this is superseded by
                                         # `max_steps` (the episode always runs to done).
    n_epochs: int = 5                   # paper: K_epoch = 5, unchanged
    clip_eps: float = 0.2               # paper: epsilon = 0.2, unchanged
    gae_lambda: float = 0.95            # paper: lambda_GAE = 0.95, unchanged
    vf_coef: float = 0.5                # paper: c_vf = 0.5, unchanged
    entropy_coef: float = 0.01          # paper: c_ent = 0.01, unchanged
    actor_lr: float = 5e-4              # paper: eta_actor = 5e-4, unchanged
    critic_lr: float = 1e-3             # paper: eta_critic = 1e-3, unchanged
    max_grad_norm: float = 0.5          # NOT in Table 1; standard PPO stabilizer, added.

    # ---------------------------------------------------------------
    # Verification & compute  (Table 1, "Verification & Compute")
    # ---------------------------------------------------------------
    seed: int = 42                      # paper: 5 seeds {42, 107, 219, 314, 501}. We default to
                                         # the first one; run train.py once per seed yourself to
                                         # reproduce the paper's 5-seed error bands.
    iterations: int = 150               # paper: 1.0e7 total environment steps over 18.4 h on an
                                         # M4 Pro with 16 SITL workers. SCALED DOWN drastically --
                                         # with max_steps=150 and episodes_per_iter=4 this default
                                         # is ~90k environment steps, a demo-scale run that
                                         # finishes in minutes, not a paper-scale training run.
    episodes_per_iter: int = 4          # see n_envs note above; this is the real "how many
                                         # episodes per PPO update" knob in this implementation.

    # ---------------------------------------------------------------
    # Kinematic-prior / PI-TGAT specific (not a Table 1 row, but is the
    # paper's core architectural mechanism -- exposed as a toggle so a
    # memoryless-GAT baseline can be obtained by setting this False)
    # ---------------------------------------------------------------
    use_kinematic_prior: bool = True    # False -> plain memoryless multi-head GAT baseline
                                         # (matches the paper's stated baseline architecture:
                                         # dropped neighbors are masked out, no decay/caching).

    # ---------------------------------------------------------------
    # Baseline toggles (NOT Table 1 rows). These, plus use_kinematic_prior
    # above, let ONE codebase produce all four architectures the paper
    # compares, as combinations of two independent axes -- aggregator type
    # and whether cross-agent communication happens at all -- rather than
    # four separately-written models:
    #   PI-TGAT (ours):  use_kinematic_prior=True,  aggregator=attention, use_graph=True  (default)
    #   TarMAC-lite:      use_kinematic_prior=False, aggregator=attention, use_graph=True  (--no-kinematic-prior)
    #   DGN-lite:         use_kinematic_prior=False, aggregator=conv,      use_graph=True  (--aggregator conv --no-kinematic-prior)
    #   Vanilla MAPPO:    use_graph=False (--no-graph; aggregator/kinematic-prior irrelevant, no graph at all)
    # See networks.py and README.md ("Baselines, combined") for the honest
    # caveats on how close each is to the paper's actual cited method.
    # ---------------------------------------------------------------
    aggregator: str = "attention"       # "attention" (GATv2Conv, learned/anisotropic) or
                                         # "conv" (isotropic mean aggregation, DGN-style)
    use_graph: bool = True              # False -> no cross-agent communication at all
                                         # (plain per-agent MLP+GRU -> Vanilla MAPPO baseline)

    # ---------------------------------------------------------------
    # Evaluation-only: forced NODE dropout (not a Table 1 training row).
    # The paper trains under organic, distance-based RF link dropout
    # (governed by r_comm/kappa above), but EVALUATES trained policies
    # under an additional, independent "simultaneous node dropout" stress
    # test -- Table 1's caption and Model Architecture section both
    # describe evaluation at up to 30% simultaneous node dropout. This
    # field defaults to 0 (no effect during training); evaluate.py sets it
    # per dropout level it's testing. When > 0, each step every agent has
    # this probability of going silent entirely (no outgoing packets that
    # step, regardless of distance to its listeners).
    # ---------------------------------------------------------------
    node_dropout_rate: float = 0.0

    device: str = "auto"                # "auto" picks mps > cuda > cpu

    def resolve_device(self):
        import torch
        if self.device != "auto":
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
