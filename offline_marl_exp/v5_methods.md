# Methods, formalised

Companion to `claude/v5_paper_outline.md` (S2 and S3). Notation is fixed here and used by every results section.
Every symbol maps to a named quantity in the code; the file and identifier are given in the margin notes so a reader
can check the implementation against the text. Defaults in brackets are the values used for all reported runs.

---

## M1. Problem setting

$N$ UAVs indexed $i \in [N]$ serve $U$ mobile ground users $u \in [U]$ over $C$ orthogonal channels for $T$ discrete
steps, in a square arena $[0,W]^2$. [$N=8$, $U=32$, $C=3$, $T=20$, $W=20$.] &nbsp;*(`commenv.EnvConfig`)*

**State.** UAV $i$ has position $p_i(t) \in [0,W]^2$ and channel $c_i(t) \in [C]$. User $u$ has position
$x_u(t)$, evolving exogenously (M6) and independent of the UAVs.

**Action.** Each UAV emits $a_i(t) \in \mathbb{R}^2$ and a channel $c_i(t{+}1) \in [C]$. The displacement is projected
onto the unit ball and scaled, so the speed limit holds by construction rather than by penalty:
$$d_i(t) \;=\; \frac{a_i(t)}{\max\{1,\|a_i(t)\|\}}, \qquad
p_i(t{+}1) \;=\; \Pi_{[0,W]^2}\!\big(p_i(t) + v_{\max}\, d_i(t)\big),$$
with $\Pi$ the componentwise clip. [$v_{\max}=1$.] &nbsp;*(`commenv.CommEnv.step`, `expert._positions`)*

## M2. Channel model

Air-to-ground gain between UAV $i$ and user $u$, with altitude $H$ entering as a path-loss floor:
$$g_{u,i}(t) \;=\; \big(\|x_u(t)-p_i(t)\|^2 + H^2\big)^{-\alpha/2}.$$
[$H=2$, $\alpha=2.5$.] Each user is served by its strongest UAV, $\sigma_u(t) = \arg\max_i g_{u,i}(t)$ (equivalently
the nearest, since $g$ is strictly decreasing in distance). Writing $S_u = g_{u,\sigma_u}$ for the signal and
$$I_u(t) \;=\; \sum_{i \neq \sigma_u(t)} g_{u,i}(t)\, \mathbb{1}\!\left[c_i(t) = c_{\sigma_u(t)}(t)\right]$$
for the aggregate co-channel interference, the spectral efficiency of user $u$ is
$$R_u(t) \;=\; \log_2\!\Big(1 + \tfrac{S_u(t)}{\eta + I_u(t)}\Big), \qquad \eta = N_0/P .$$
[$\eta = 4\times10^{-4}$, transmit power normalised to 1.] &nbsp;*(`commenv.link_rates`)*

**Why interference is long-range here.** $I_u$ sums over *all* co-channel UAVs with no distance cut-off, and
$\alpha = 2.5$ makes the tail heavy enough that distant co-channel UAVs are not negligible. This is the mechanism
behind the observability result (outline S4) and it should be stated in the setting, not discovered in the results.

## M3. Objective

The team reward is utilitarian in rate and charges propulsion energy quadratically:
$$r(t) \;=\; \frac{1}{U}\sum_{u=1}^{U} R_u(t) \;-\; \lambda\,\frac{1}{N}\sum_{i=1}^{N} \|d_i(t)\|^2,
\qquad G \;=\; \sum_{t=1}^{T} r(t).$$
[$\lambda = 0.5$.] &nbsp;*(`commenv.CommEnv.step`)*

**Two coverage statistics are reported but never optimised** (except where M9 says otherwise):
$$\Omega \;=\; \frac{1}{UT}\sum_{t,u} \mathbb{1}[R_u(t) < 1], \qquad
Q_{10} \;=\; \text{10th percentile of } \{R_u(t)\}_{u,t}.$$
$\Omega$ is the outage fraction at a fixed 1 bit/s/Hz threshold. **The threshold is fixed at 1 in every table**, so the
statistic is comparable across arms even when a training-time floor $\rho \neq 1$ is used. &nbsp;*(`commenv` `info["outage"]`)*

Note explicitly in the paper: $r$ has no floor on the worst-served user. Outline S7 is the consequence.

## M4. Observation: the $k$-masked set observation

Agent $i$ observes a tuple of a self-vector and two variable-length sets, truncated by radius:
$$o_i^{k,k_a}(t) \;=\; \Big(\; s_i(t),\; \{\,f_{u,i}(t)\,\}_{u \in \mathcal{U}_i^{k}},\; \{\,h_{j,i}(t)\,\}_{j \in \mathcal{N}_i^{k_a}} \Big)$$
$$\mathcal{U}_i^{k} = \{u : \|x_u - p_i\| \le k\}, \qquad
\mathcal{N}_i^{k_a} = \{j \neq i : \|p_j - p_i\| \le k_a\}$$
with components (all lengths normalised by $W$ or $W/2$ so features are $O(1)$):
$$s_i = \big(\; p_i/W,\;\; \mathrm{onehot}(c_i),\;\; t/T,\;\; \beta_i \;\big), \qquad
f_{u,i} = \Big(\tfrac{x_u - p_i}{W/2},\; \tfrac{\dot{x}_u}{v_{\max}}\Big), \qquad
h_{j,i} = \Big(\tfrac{p_j - p_i}{W/2},\; \mathrm{onehot}(c_j)\Big).$$
&nbsp;*(`commenv.CommEnv.obs`, returning `Obs(sf, uf, um, af, am)`)*

$k$ and $k_a$ are set **independently**: $k$ truncates the user set, $k_a$ the UAV set. Separating them is what isolates
"distant users" from "distant UAVs" in outline S4; $k_a = k$ unless stated.

$\beta_i \in \mathbb{R}^{m}$ is the optional global-summary block, defined next. $m=0$ for the plain $k$-masked actor.

## M5. The two proposed features (and their controls)

All variants are **fixed-width and rotation-invariant**, so they are global *summaries*, not entity lists, and they do
not break the augmentation of M8. &nbsp;*(`commenv.CommEnv.obs`, `EnvConfig.bcast`)*

**(a) Per-channel broadcast count** — one integer per channel per UAV per step, the communication cost of the scheme:
$$\beta^{\mathrm{cnt}}_i[c] \;=\; \frac{1}{N}\,\big|\{\, j \neq i \;:\; c_j = c \,\}\big| \;\in \mathbb{R}^{C}. \qquad [\texttt{bcast}=1]$$

**(b) Receiver-side co-channel power** — requires no communication at all:
$$\beta^{\mathrm{sens}}_i[c] \;=\; \frac{1}{\kappa}\,\log\!\Big(1 + \tfrac{1}{\eta}\textstyle\sum_{j\neq i} g_{i,j}\,\mathbb{1}[c_j = c]\Big),
\qquad g_{i,j} = \big(\|p_j-p_i\|^2+H^2\big)^{-\alpha/2}, \qquad [\kappa = 8,\;\texttt{bcast}=3]$$
i.e. the same path-loss kernel as the user link, evaluated at UAV $i$'s own position. **Name it precisely**: this is the
co-channel power a receiver co-located with UAV $i$ would measure, which is what its own served users approximately
experience — receiver-side interference feedback, not UAV-side spectrum sensing. It is *physically local* but
*informationally global*, which is exactly why it works.

**Controls, needed for the claim to be causal:**
- **Radius-restricted** $\beta^{\mathrm{sens}}$ with the sum over $j \in \mathcal{N}_i^{k_a}$ only [$\texttt{bcast}=4$] —
  tests whether the value is in the long-range tail rather than the near field.
- **Placebo** $\pi_{i,t}\big(\beta^{\mathrm{sens}}_i\big)$ [$\texttt{bcast}=5$], where $\pi_{i,t}$ is a deterministic
  pseudorandom permutation of the $C$ entries depending only on (scenario seed, $i$, $t$). Identical marginal
  distribution and identical input width, zero information about *which* channel is quiet — so a gain over this
  placebo cannot be an effect of feature dimensionality.
- **Combined** $\beta_i = [\beta^{\mathrm{cnt}}_i ; \beta^{\mathrm{sens}}_i] \in \mathbb{R}^{2C}$ [$\texttt{bcast}=6$].

*(A fourth variant, per-UAV user-distance-ring shares [$\texttt{bcast}=2$], is reported as a null.)*

## M6. Scenario generation

A scenario is a deterministic function of its integer seed. [4 Gaussian clusters, $\sigma=2$, cluster drift speed
$\le 0.4$ per step, reflecting boundaries; UAV positions and channels uniform.] &nbsp;*(`commenv.gen_scenario`)*
This makes paired evaluation exact: every arm is rolled out on bit-identical scenarios (common random numbers).

## M7. Policy architecture

The same encoder serves actor and critic; only the attention mask differs (M10).
With a learned always-visible null token $z_0$ (so the softmax is never empty when a set is):
$$q_i = \mathrm{MLP}_s(s_i),\quad e_u = \mathrm{MLP}_u(f_{u,i}),\quad e_j = \mathrm{MLP}_a(h_{j,i}),$$
$$\mathrm{ctx}_i = \mathrm{MHA}\big(q_i,\; K_i,\; K_i\big),\quad K_i = [\,z_0;\{e_u\}_{u\in\mathcal{U}_i^k};\{e_j\}_{j\in\mathcal{N}_i^{k_a}}\,],$$
$$\textstyle\Sigma^{\mathcal U}_i = \frac{1}{8}\sum_{u\in\mathcal{U}_i^k} e_u, \qquad
\Sigma^{\mathcal N}_i = \frac{1}{4}\sum_{j\in\mathcal{N}_i^{k_a}} e_j, \qquad
z_i = \mathrm{MLP}_o\big([\,q_i;\mathrm{ctx}_i;\Sigma^{\mathcal U}_i;\Sigma^{\mathcal N}_i\,]\big).$$
The sum-pooled terms use **fixed** normalisers, not the set size, so entity *counts* survive the pooling — attention
alone would discard them. [$d = 128$, 4 heads.] &nbsp;*(`models.SetEncoder`)*

**Heads** *(`models.StudentActor`)*:
- **Position**, a mixture density with $K$ components, so a multimodal teacher is not averaged into a hovering UAV:
  $(\mu_k, \log\varsigma_k, w_k)_{k=1}^{K} = \mathrm{Linear}(z_i)$, and the deterministic action is $\mu_{k^\star}$ with
  $k^\star = \arg\max_k w_k$. [$K=5$.]
- **Channel**: categorical logits $= \mathrm{Linear}(z_i)$.
- **Exploration**: a state-independent $\log\varsigma$ used only online (M10).
- **Auxiliary chunk head** predicting the planner's next $H-1$ displacements (representation shaping; never executed).
  [$H=4$.]

## M8. Teacher: receding-horizon block-coordinate descent

Over a remaining horizon $h$, the planner maximises the **environment return itself** — the same functional, not a
surrogate (verified by replay):
$$\max_{u,\,c}\; J(u,c) = \sum_{t=1}^{h}\Big[\tfrac{1}{U}\textstyle\sum_u R_u(t) - \lambda \tfrac{1}{N}\sum_i \|d_i(t)\|^2\Big]
\quad\text{s.t.}\quad d = \tfrac{u}{\max(1,\|u\|)},\;\; p(t{+}1)=\Pi\big(p(t)+v_{\max}d(t)\big).$$
*(`expert._objective`, `planner_state.plan_from_state`)*

This is a non-convex mixed-integer problem (continuous trajectories coupled to discrete channels through SINR). BCD:
- **Block A — trajectories given channels**: Adam on $u$ with cosine-annealed LR, retaining the best iterate, which
  makes the block monotone. [$\text{lr}=0.08$.]
- **Block B — channels given trajectories**: exact coordinate-wise best response. Given positions the objective is
  separable over time, so each $(i,t)$ is an exhaustive $C$-way search; 2 sweeps over $i$.
- **Restarts**: restart 0 is a deterministic warm start (farthest-point init + Lloyd iterations + Hungarian assignment
  of UAVs to cluster centroids); restart 1 perturbs it; the rest are random. The arg-max restart is taken, preferring
  the warm start when within 1% (label persistence).
- **Label canonicalisation**: the reward depends only on *which* UAVs share a channel, never on the label, so channel
  labels are permuted per step to agree maximally with the previous step. This removes pure labelling noise from the
  imitation target and leaves $J$ bit-for-bit unchanged.

**State this prominently.** The planner is a strong heuristic, **not a certified optimum**, and it does not converge
with budget (outline S3.2). Consequently every normalised score names its reference, and the paper's contrasts are
reported at the return level, where the reference cancels.

## M9. Distillation: BC then DAgger

Let $\pi_\theta$ be the student and $\bar\pi_\theta$ its EMA. &nbsp;*(`student.py`, `bc2.py`)*
- **Round 0 (behaviour cloning).** The planner solves $n_0$ scenarios open-loop over the full horizon; the plans are
  *replayed through the environment* so the dataset carries the student's own observations rather than reconstructed
  ones. [$n_0 = 600$.]
- **Rounds $r = 1..R$ (DAgger).** Roll $\bar\pi_\theta$ out on fresh scenarios with exploration noise
  $\epsilon \sim \mathcal{N}(0,\varsigma_{\mathrm{ex}}^2)$. **At every visited state** the planner is restarted from
  that state over the remaining horizon, and its first action becomes the label. With probability $\beta_r = \beta_0^r$
  the planner's action is executed instead of the student's (the DAgger mixture), so the state distribution anneals
  from the teacher's to the student's. [$R=8$, 256 scenarios/round, $\beta_0 = 0.5$, $\varsigma_{\mathrm{ex}}=0.05$.]
- **Loss.** With $a^\star$ the labelled displacement and $c^\star$ the labelled channel:
  $$\mathcal{L} = \underbrace{-\log \textstyle\sum_k w_k\,\mathcal{N}(a^\star;\mu_k,\varsigma_k)}_{\text{MDN NLL}}
  \;+\; w_{\mathrm{ch}}\,\mathrm{CE}(\text{logits}, c^\star)
  \;+\; w_{\mathrm{ck}}\,\big\|\hat{A}_{2:H} - A^\star_{2:H}\big\|^2_{\text{masked}} .$$
  [$w_{\mathrm{ch}}=1$, $w_{\mathrm{ck}}=0.5$.]
- **Dihedral augmentation.** The physics is invariant under $D_8$, the symmetry group of the square (isotropic path
  loss, square arena, reflecting boundaries). Observation/action pairs are transformed by a random element, an 8×
  data multiplier. Every feature in M4–M5 is equivariant or invariant by construction, which is what makes the sensing
  feature of M5(b) safe to add. *(`bc2.augment2`; verified by a unit test.)*
- **Model selection.** Round selected by closed-loop return on the **monitor** seed set only (M11).

## M10. Online fine-tuning: MAPPO

Parameter-shared actor, factored policy $\pi_\theta(a,c\mid o^k) = \mathcal{N}(a;\mu_\theta(o^k), \varsigma^2 I)\cdot
\mathrm{Cat}(c;\text{logits}_\theta(o^k))$ with state-independent $\varsigma$ initialised to $\varsigma_0$, and a
centralised critic $V_\phi$. Clipped surrogate with GAE and approximate-KL early stopping:
$$\mathcal{L}^{\mathrm{actor}} = -\,\mathbb{E}\Big[\min\big(\varrho\,\hat{A},\;\mathrm{clip}(\varrho,1\pm\varepsilon)\hat{A}\big)\Big]
- c_{\mathrm{ent}}\mathbb{E}[\mathcal{H}], \qquad \varrho = \tfrac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}} .$$
[$\varepsilon = 0.2$, $\gamma = 0.99$, $\lambda_{\mathrm{GAE}} = 0.95$, $c_{\mathrm{ent}} = 10^{-3}$, reward scale
$0.05$, target KL $0.03$ with early stop at $1.5\times$, 3 epochs, 2 minibatches, lr $3\times10^{-4}$ actor /
$10^{-3}$ critic, 32 envs $\times$ 20 steps $=$ **640 team steps per iteration**, $\varsigma_0 = 0.2$, critic warm-up
100 iterations with the actor frozen.] &nbsp;*(`mappo.py`, `online2.py`)*

**The critic-mask ablation.** The masked critic consumes the actor's own $k$-masks; the unmasked critic consumes full
masks. Same class, same architecture, same initialisation under the same seed — **the attention mask is the single
differing factor**. This is the design that makes the null in outline S8 interpretable.

### M10.1 The coverage constraint
A penalty is subtracted from the **training reward only**:
$$\tilde{r}(t) \;=\; r(t) \;-\; c_{\Omega}\,\Omega(t) \;-\; c_{\Phi}\,\Phi_\rho(t), \qquad
\Phi_\rho(t) = \frac{1}{U}\sum_u \max\{0,\; \rho - R_u(t)\}.$$
Evaluation, the logged episode return, and every held-out number use $r$, never $\tilde{r}$; and the reported $\Omega$
keeps its fixed threshold of 1 whatever $\rho$ is. Constrained and unconstrained runs are therefore directly
comparable, and $c_\Omega = c_\Phi = 0$ reproduces the unconstrained runs exactly. &nbsp;*(`mappo.PPOConfig.outage_coef`,
`shortfall_coef`, `EnvConfig.rate_floor`; two unit tests enforce both properties.)*

$\Omega$ is a step function of the rate and $\Phi_\rho$ a graded hinge. The empirical finding (outline S7.3) is that
**the floor must satisfy $\rho >$ the reporting threshold** for the graded form to behave differently from the step
form: at $\rho = 1$ both target the same users.

## M11. Evaluation protocol

**Three disjoint seed sets, never mixed** *(`final_eval.py --seed0/--tag`)*:
monitor/selection $7\!\times\!10^5{+}$ (chooses the DAgger round and PPO monitor evals), development
$8\!\times\!10^5..8\!\times\!10^5{+}255$, and a **frozen test set** $9\!\times\!10^5..9\!\times\!10^5{+}1023$ used once
per headline number.

**Normalised score.** With $\bar{G}_{\mathrm{stay}}$ the do-nothing policy and $\bar{G}_{\mathrm{exp}}$ the planner on
the *same* scenarios,
$$\mathrm{NS} \;=\; \frac{\bar{G}_{\pi} - \bar{G}_{\mathrm{stay}}}{\bar{G}_{\mathrm{exp}} - \bar{G}_{\mathrm{stay}}}.$$
The denominator depends on the planner budget (M8) and on the seed set, so **every normalised number names both**, and
normalised scores are not compared across seed sets. Return-level contrasts are reference-free and are the primary
reporting unit. *(`evalproto.expert_normalised`; references cached per (seed set, planner) in `results2/final/refs_*.npz`.)*

**Unit of analysis.** The independent replicate is the **offline student** (an independent BC+DAgger run), not the PPO
seed: students of the same configuration share no weights, whereas PPO seeds of one student do. CIs are $t$-based over
students. [3 students per configuration.] A variance decomposition of the paired critic-mask contrasts shows that at a
fixed run budget, moving replicates from PPO seeds onto students tightens the interval by about $2.4\times$ — stated in
the paper as a design recommendation we did not have the budget to adopt.

**Paired evaluation** throughout: identical scenario seeds for every arm, so between-arm differences are not inflated
by scenario variance.

---

## Reporting conventions to state once and apply everywhere
1. Every normalised score names its planner budget and seed set.
2. Every online table reports a coverage statistic ($\Omega$ or $Q_{10}$) beside the return — the return alone makes
   from-scratch RL and long fine-tuning look better than they are (outline S6.2).
3. CIs are over offline students, $t$-based, $n=3$ unless stated.
4. "Team step" = one environment step of one scenario; one PPO iteration = 640 team steps. Sample budgets are quoted
   in team steps, and the offline stage is counted (outline S6.1).
