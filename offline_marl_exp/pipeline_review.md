# Code and results review: Range-Bounded Asymmetric Distillation pipeline

I ran your code. Every number below is measured, not inferred — the diagnostic cells are reproducible from
the attached files plus `eval_protocol.py`.

![Diagnostics]({{artifact:art_2365c8cb-ebe6-454a-8d93-5a2dd431df22}})

**Bottom line.** The research framing is sound and still worth doing. The current pipeline cannot support
any claim: the online policy never updates (gradient is identically zero), two of the three arms in the
plot are the *same frozen network*, and the distilled student performs worse than a policy that does
nothing. The plotted differences are scenario noise. None of this is fatal to the idea — all of it is
fixable — but nothing from this run can go in a paper, including as a preliminary result.

---

## 1. Fatal defects (they invalidate the plotted result)

**F1 — The actor receives exactly zero gradient; the "fine-tuning" updates nothing.**
In `phase3_online_finetuning.py` and `evaluate_baselines.py`, actions are drawn from the actor under
`torch.no_grad()`, then the surrogate is `-Σ(new_pos - pos_actions)²` with `new_pos = actor(same_state)`.
Since the actor is deterministic, `new_pos == pos_actions` exactly, so the loss and its gradient are both
zero. Measured: `actor_loss = 0.000e+00`, total gradient norm `0.000e+00`, and `max|Δweight| = 0.000e+00`
after three epochs of fine-tuning. The channel head gets `grad = None` — the discrete action is never
trained online at all, because `channel_logits` never enters any loss.

**F2 — "Baseline C" and "Ours" are the same network.** Both are `copy.deepcopy(pretrained_actor)`, and by
F1 neither changes. The only difference is the critic, which by F1 cannot influence the actor. The true
effect size between those two curves is exactly zero; everything visible in the figure is
scenario-to-scenario variance from `env.reset()`.

**F3 — The behaviour-cloning target is one of the input features.** In `bc_masked_actor.py`,
`state_features.extend(own_pos)` makes `own_pos` dimensions 0–1 of the observation, and
`target_pos = own_pos`. The student is trained to copy two inputs to its output: the task carries no
information about the expert's decision. Consequences, measured over 300 held-out scenarios
(mean episode return, higher is better):

| policy | return |
|---|---|
| expert solver (oracle) | **−521 ± 164** |
| do nothing (stay put) | −755 ± 288 |
| random targets | −937 ± 177 |
| **shipped BC student ("Ours")** | **−1944 ± 478** |

Paired against stay-put over 120 common scenarios, the student is **−1165 return worse, 95 % CI
[−1252, −1079]**. It is not a weak policy, it is an actively harmful one, and it flies at the speed cap
(mean step 0.52 of a 0.5 cap) while doing it.

**F4 — The decision transformer is given the action it is asked to predict.** In `madt.py`, the action
window `a_pos_seq[:, -1]` is the target `target_pos` (verified: `torch.allclose(...) == True`), and the
prediction head reads the last token, which embeds it. Any offline loss you report is label leakage. For
the record the shipped checkpoint has not even exploited it — position MSE 6.79 with the action token
present, 6.92 with it zeroed, against a target variance of 8.92, i.e. barely better than predicting the
mean. Also, the model adds the return-to-go, state and action embeddings into a single token rather than
interleaving `[R̂, s, a]` as in Decision Transformer; with the sum you cannot condition on return at
inference in the usual way.

**F5 — Offline states are not the states the policy will see.** `madt.py` builds observations from
`u_data['action_pos']` ("using pos target as proxy for own pos for simplicity"), not from the recorded
`obs`. Measured mismatch against the env's own observation at step 0: mean |difference| = **3.36** in a
10-unit world. You are training on one observation distribution and deploying on another.

**F6 — Dataset provenance is broken.** `bc_masked_actor.py` loads `data/expert_dataset_sample.json` and
reads key `optimal_uavs`; `expert_solver.py` writes `data/expert_sequence_dataset_sample.json` with key
`optimal_target_uavs`. The repo as shipped cannot regenerate the file the saved checkpoint was trained on.

## 2. Serious defects (they invalidate the claim even once F1–F6 are fixed)

**S1 — There is no trust region anywhere.** No probability ratio, no clipping or KL constraint, no old
policy, no GAE or discounted returns (the critic regresses `V(s_t)` onto the *immediate* reward), no
entropy term, no exploration noise, no minibatching or multiple epochs per batch. A paper whose thesis is
"trust-region fine-tuning with a masked critic" must actually run HAPPO/MAPPO. Use an existing
implementation (HARL, arXiv:2304.09870) rather than writing the update by hand — the hand-written version
is where F1 came from.

**S2 — The evaluation measures the wrong thing, with no power.** The plot is per-epoch *training* reward,
one episode per point, one seed, on a freshly randomised scenario each time, unpaired across arms. Given
the measured σ ≈ 478, that design resolves only a **22 % effect** at 80 % power. To resolve 5 % you need
≈ 380 episodes per arm unpaired, or ≈ 181 paired on common scenario seeds (panel b). There is also no
oracle line and no trivial floor, which is why F3 went unnoticed for so long.

**S3 — Arms differ in two things at once.** "Baseline A" changes both initialisation *and* critic type
relative to "Ours". Vary one factor per arm.

## 3. Design defects (they make the central claim untestable in this environment)

**D1 — The mask does not hide anything the reward depends on.** Interference is penalised only for
same-channel pairs closer than 4.0, and `k_radius = 5.0`. Measured over 300 scenarios: **100 %** of
reward-coupling pairs (315/315) lie *inside* the mask. So for the only inter-agent coupling term in your
reward, L(k) ≡ 0 by construction, and a masked critic cannot differ from an unmasked one on it. Separately,
the observation caps visible users at 5 and that cap binds in **35 %** of agent-steps, so a k-ablation
varies two bottlenecks at once. As it stands the environment cannot produce the curve your theorem
predicts. Fix: set the interference radius well *beyond* the smallest k you sweep, sweep k across it, and
remove the fixed entity cap (use attention over a variable-length set).

**D2 — The expert is not a MINLP solver and there is no sequential problem to distil.**
`scipy.optimize.differential_evolution` is a stochastic global heuristic with no optimality certificate;
you report `res.fun` with no gap. Worse, it solves a *static placement*, after which the "expert
trajectory" repeats one constant target for all 20 steps. A Decision Transformer over a constant action
sequence has nothing to model. Either make the problem genuinely sequential (user mobility, time-varying
demand, energy budget, so the optimal plan is a trajectory) or drop the sequence model and call it
distillation of a one-shot optimiser.

**D3 — The environment is a facility-location toy, not a communications system.** No SINR, no transmit
power, no bandwidth, no path loss, no user association, no outage. "Coverage cost = Σ min-distance" and a
hinge penalty are not a wireless model, and reviewers at a comms venue will reject on this alone. You need
at minimum: log-distance or 3GPP A2G path loss, SINR with aggregate interference from all co-channel UAVs,
Shannon rate, and a per-user QoS or sum-rate objective.

**D4 — Observation encoding.** Zero-padding means "entity at my exact position"; the visible-user list is
unsorted, so the MLP is permutation-sensitive; features are unnormalised absolute coordinates. Use
relative coordinates, sort or (better) attend over the variable-length set with an explicit presence flag.

## 4. The theorem needs repair before it carries weight

The statement in `gap_bound_theorem.md` is the right shape but not yet a theorem:

1. **Dimensions.** The left side is in return units; the right side is a divergence plus an unspecified
   L(k). A TV-based bound needs a reward scale: the standard form is `2γ R_max/(1−γ)² · ε`. As written the
   inequality is not dimensionally consistent.
2. **L(k) is circular.** It is defined as "the maximum expected impact on the optimal action choice caused
   by state variables outside radius k" — that is the thing to be bounded, not a primitive. Define it from
   the model: a coupling coefficient between agent *i*'s value and the state of agents at graph distance
   > k (the Dobrushin-style row sums in arXiv:2602.16966, or the exponential-decay property of
   arXiv:2006.06626), then *prove* the value gap in terms of it.
3. **The O(1/k^α) claim is about the wrong quantity.** Per-source interference power decaying as d^−α does
   not give value impact decaying as k^−α: you must bound the *aggregate* over all sources beyond k, and
   in a 2-D field with α ≤ 2 that aggregate does not converge. This is exactly the objection I flagged in
   the first audit, and it is also where the paper's real theoretical content lives — characterising when
   the aggregate coupling is summable is the contribution.
4. **Behaviour cloning from a privileged expert is not bounded by per-state TV.** Under masking the student
   is in a Dec-POMDP where the teacher's policy may be *unrealisable*, and the imitation gap can be
   arbitrarily large regardless of how well you fit the data — the standard references are
   arXiv:2007.12173 (ADVISOR), arXiv:1912.12294 (Learning by Cheating), arXiv:1710.06542 (asymmetric
   actor-critic), and the two that matter most for your statement: **arXiv:2505.09546** (Distilling
   Realizable Students from Unrealizable Teachers) and **arXiv:2412.00985** (Provable Partially Observable
   RL with Privileged Information). Your bound must either assume realisability within the masked class or
   carry an unrealisability term.
5. **Horizon term.** BC's compounding error is O(T²ε), DAgger's O(Tε) (arXiv:1011.0686). A bound with no
   horizon factor will be challenged immediately; it also motivates the online phase properly.
6. **Add a tightness example.** A two-agent instance where the bound is attained turns a plausible
   inequality into a result.

## 5. Rebuild order

1. **Sanity gates in CI first** (`eval_protocol.py` → `sanity_gates`). Non-zero gradient on every head,
   weights change, arms are distinct networks, policy beats stay-put and random. These four assertions
   would have caught F1–F3 on day one. They currently fail at gate 1.
2. **Fix the BC target**: predict the expert's *displacement* (or next waypoint) from the masked
   observation, never a quantity present in the input. Re-check: student must beat stay-put by a wide,
   CI-separated margin before anything else is worth running.
3. **Rebuild the environment** with a real channel model (D3), interference radius > max k (D1), variable
   user demand or mobility so the problem is sequential (D2), and set-based observations (D4).
4. **Replace the expert**: SCA/BCD with a reported optimality gap, or a MINLP solver on a convexified
   subproblem; log solve time and the certificate. Keep the heuristic only as a secondary "weak expert".
5. **Use a real HAPPO/MAPPO implementation** for phase 3, with the only modification being the critic's
   attention mask. The mask must be the *single* difference between "Ours" and "unmasked critic".
6. **Re-derive the theorem** per §4, and estimate L(k) empirically in the new environment (perturb an
   agent beyond radius k, measure the induced change in another agent's return — that curve, overlaid with
   the theoretical bound, is your Figure 2).
7. **Then** run the comparison, paired, with the sample sizes from §6.

## 6. Evaluation protocol to adopt now

- **Paired scenarios (common random numbers).** Every arm sees the identical seed set. Measured here:
  pairing cuts the required episodes roughly in half (181 vs 380 for a 5 % effect).
- **Separate training from evaluation.** Report deterministic evaluation on a held-out seed set at fixed
  checkpoints, not per-epoch training reward.
- **≥ 5 training seeds per arm**, with the seed as the unit of analysis; shade 95 % bootstrap CIs over
  seeds, not over episodes within a seed.
- **Always plot three reference lines:** expert/oracle, do-nothing, random. Report the expert-normalised
  score (student − do-nothing)/(expert − do-nothing); `paired_eval` computes it.
- **Pre-register the effect size you intend to detect** and size the run accordingly (panel b).
- **Ablations that matter:** k ∈ {1, 2, 3, ∞} with the interference radius fixed inside the sweep; masked
  vs unmasked critic with everything else identical; BC-only vs BC + online vs online-only; expert quality
  (strong vs weak) at fixed data volume.

## 7. What the paper can claim, in order of what you can support

1. *(after step 2)* a decentralised student distilled from a centralised optimiser reaches X % of oracle
   return at Y ms inference versus the solver's Z s — the speed/quality trade-off, which is the
   comms-venue story;
2. *(after steps 3–6)* measured L(k) versus the bound: the value gap grows as the mask tightens, and the
   bound tracks it — this is the theory contribution and the reason the paper is not just engineering;
3. *(after step 7)* the masked critic closes the residual gap faster than an unmasked critic, with
   CI-separated curves over ≥ 5 seeds.

Claim 1 alone, done properly, is already a publishable comms paper. Claim 2 is what differentiates you
from arXiv:2608.28878. Claim 3 is the weakest and should be reported honestly even if the effect is small
or negative — "the mask costs nothing and buys scalability" is a perfectly good result, and much safer
than an unsupported superiority claim.

---

### Files
- `eval_protocol.py` — paired evaluation with bootstrap CIs, expert-normalised scores, minimum-detectable-
  effect calculator, and the four sanity gates. Drop it next to `env.py`; it imports nothing of yours but
  `DroneSwarmEnv`.
- `pipeline_diagnostics.png` — panel a: return distributions (n = 300 scenarios; expert n = 40);
  panel b: minimum detectable effect versus episodes, paired and unpaired.
