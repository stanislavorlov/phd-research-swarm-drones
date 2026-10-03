# Performance-gap analysis for range-bounded distillation — v2 (replaces `gap_bound_theorem.md`)

Status legend: **[proved]** = proof given below · **[measured]** = estimated in `measure_Lk.py` · **[open]** = not claimed.
Nothing here depends on the neural networks; everything is stated for the environment in `commenv.py`.

## 0. Setting and the one structural fact that matters

Finite horizon `T`, `N` UAVs, `U` users, per-step team reward

`r(s) = (1/U) Σ_u log2(1+SINR_u) − λ_E (1/N) Σ_i ‖ψ(a_i)‖²`, `SINR_u = S_u / (N0 + I_u)`,
`S_u = g(d_{u,s(u)})`, `I_u = Σ_{j≠s(u), c_j=c_{s(u)}} g(d_{uj})`, `g(d) = (d²+H²)^{−α/2}`, `s(u)` = nearest UAV.

**Structural fact (S).** User trajectories are exogenous and UAV motion is decoupled across agents
(`p_i' = clip(p_i + v_max ψ(a_i))`). *The only coupling between agents is the reward, through interference.*
Hence changing the reward changes no trajectory law — every policy induces the same state distribution under `r`
and under any other reward function. All results below use this.

`J(π) = E_π Σ_{t≤T} r(s_t)`. `Π_k` = decentralised policies `a_i ∼ π_i(·|o_i^k)` where `o_i^k` contains only entities
within horizontal distance `k` of UAV `i`. `π*` = full-information teacher. `R̄ :=` the range (max − min) of `r`, so every
bound below is in **return units** (this fixes the dimensional error in v1).

## 1. Defining `L(k)` from the model (v1 defined it as "the impact of the far field", which is circular)

**k-local reward.** `r^{(k)}(s)` is `r(s)` with, for every user `u`, all interferers `j` with `‖p_j − p_{s(u)}‖ > k` removed
from `I_u` (they are outside what the serving UAV can observe). Because removing interference can only raise a rate,

`ε_r(k) := sup_s ( r^{(k)}(s) − r(s) ) ≥ 0`, and `ε_r(k) → 0` as `k → diam(world)` (nothing lies beyond `k`).

`ε_r` is a primitive of the *model* (a property of `g`, `N0` and the geometry), not of any policy.

## 2. Results

### Proposition 1 — rate perturbation **[proved]**
For `S>0`, `X ≥ N0 > 0`, `Y ≥ 0`:
`0 ≤ log2(1+S/X) − log2(1+S/(X+Y)) ≤ log2(1 + Y/X) ≤ log2(1 + Y/N0)`.

*Proof.* `log2(1+S/X) − log2(1+S/(X+Y)) = log2((X+Y)/X) − log2((X+Y+S)/(X+S))`. The second term is ≥ 0 (since `(X+Y+S)/(X+S) ≥ 1`),
so the difference is ≤ `log2((X+Y)/X) = log2(1+Y/X)`. Non-negativity is monotonicity of the rate in `I`. ∎
*Tight* as `S → ∞` (second term → 0): see §3.

### Proposition 2 — aggregate far-field interference **[proved]**
Let `ρ ≥ sup_u ‖x_u − p_{s(u)}‖` (service radius) and `k > ρ`. For user `u`, every dropped interferer has `‖p_j − x_u‖ > k − ρ`, so
(a) **finite network:** `I_far ≤ (N−1) · ḡ(k−ρ)`, `ḡ(x) = (x₊²+H²)^{−α/2}`;
(b) **density condition:** if every disc satisfies `#{j : ‖p_j − x‖ ≤ r} ≤ λ̄πr² + c`, then
`I_far ≤ ∫_{k−ρ}^∞ (−g′(r)) (λ̄πr² + c) dr = O((k−ρ)^{2−α})`, which is **finite iff α > 2**.

*Proof of (b).* `g(d_j) = ∫_{d_j}^∞ (−g′(r)) dr`, so by Fubini `Σ_{d_j>R} g(d_j) = ∫_R^∞ (−g′(r)) #{R<d_j≤r} dr`. Insert the density bound;
`−g′(r) = α r (r²+H²)^{−α/2−1}`, so the integrand is `~ r^{1−α}`, integrable at infinity iff `α>2`. ∎

**Consequence (the real content the review pointed at).** In a fixed-density, growing network the far field is *summable* — a bounded
`k` suffices uniformly in `N` — **iff `α>2`**. For `α ≤ 2` the aggregate beyond any `k` diverges with network size, and no
fixed observation radius controls the loss. (The v1 claim `L(k)=O(1/k^α)` bounded the per-source power, not the aggregate, and
is wrong in that regime; the correct exponent for the aggregate is `k^{2−α}`.)

Combining Props 1–2: `ε_r(k) ≤ log2(1 + Î(k)/N0)` with `Î(k)` the bound in (a) or (b). **[proved]** — compared with the measured
`ε̂_r(k)` in `results/Lk_vs_bound.png` **[measured]**.

### Theorem A — far-field blind optimisation within the masked class **[proved]**
(i) For **every** policy `π`: `|J(π) − J^{(k)}(π)| ≤ T ε_r(k)`.
(ii) If `π^{(k)} ∈ argmax_{π∈Π_k} J^{(k)}(π)` then `J(π^{(k)}) ≥ sup_{π∈Π_k} J(π) − 2 T ε_r(k)`.

*Proof.* (i) By (S) the two objectives are expectations of different per-step rewards under the **same** trajectory law; the per-step
difference is ≤ `ε_r(k)`; sum over `T` steps. (ii) For any `π ∈ Π_k`: `J(π^{(k)}) ≥ J^{(k)}(π^{(k)}) − Tε_r ≥ J^{(k)}(π) − Tε_r ≥ J(π) − 2Tε_r`. ∎
This is the clean statement of what a *k-local* learner (including a k-masked critic that models only locally observable return) can and
cannot lose through the far field: at most `2Tε_r(k)`, and `ε_r` is controlled by Props 1–2.

### Proposition 3 — imitation from a privileged teacher, with horizon and unrealisability **[proved]**
Let the joint-action total variation at step `t` be `ε_t := E_{s_t∼d_t^{π*}} TV(π*(·|s_t), π̂(·|o_t^k))` and `ε̄ = max_t ε_t`. Then

`J(π*) − J(π̂) ≤ R̄ Σ_{t=1}^{T} Σ_{s=1}^{t} ε_s ≤ R̄ · T(T+1)/2 · ε̄`      (behaviour cloning: **quadratic in T**).

*Proof.* Couple the two trajectories using a maximal coupling of the actions at each step while they agree. The probability that they have
separated by step `t` is ≤ `Σ_{s≤t} ε_s`; on the event that they have not separated the rewards coincide, otherwise they differ by ≤ `R̄`. Sum over `t`. ∎

**Unrealisability term.** Let `π̄_k(·|o) := E[π*(·|s) | o_t^k]` (posterior mixture of the teacher given what the student can see) and
`ε_info(k) := max_t E TV(π*, π̄_k)`, `ε_approx := max_t E TV(π̄_k, π̂)`. Triangle inequality: `ε̄ ≤ ε_approx + ε_info(k)`.
`ε_info(k) = 0` exactly when the teacher is realisable in `Π_k` at the visited states; otherwise it is a floor **no amount of
data or training removes** (ADVISOR, arXiv:2007.12173; Learning-by-Cheating, 1912.12294; Distilling Realizable Students from
Unrealizable Teachers, 2505.09546 — references taken from the review; I have not re-verified them). For a joint policy of `N` independent
agents, `TV_joint ≤ Σ_i TV_i`.

**DAgger-type horizon.** If the student is trained on its *own* state distribution with teacher labels (DAgger, arXiv:1011.0686) the
horizon dependence drops to `O(T)` under a bounded teacher Q-gap. Not implemented here **[open]**.

### What is **not** claimed **[open]**
* A bound of `ε_info(k)` by `ε_r(k)`. `ε_info` is a property of the *teacher's* sensitivity to far-field state, `ε_r` of the *reward's*.
  They are different objects; v1 conflated them. `ε_info(k)` is estimable: the held-out BC loss floor of a large masked student
  as `k` varies (not yet measured here).
* That `T(T+1)/2` is attained in this environment. It is not obviously: UAV states are recoverable (a deviation can be undone at speed `v_max`),
  which is exactly the regime where the quadratic compounding of behaviour cloning is pessimistic. Treat it as an upper bound.
* Anything about the online phase beyond Theorem A (the masked-critic claim is an empirical question; see `README_v2.md`).

## 3. Tightness example (for Prop. 1 and for the information gap)

Two UAVs at distance `D > k`, each with one user directly beneath it (`S = g(0) = H^{−α}`), two channels, `T = 1`, and observations that
do not distinguish the agents (exchangeable; e.g. no absolute position). The teacher puts them on different channels: per-user rate
`log2(1 + S/N0)`. Any `Π_k` policy cannot see the other UAV, and by exchangeability both play the same (possibly randomised) rule; the
deterministic case collides with probability 1, giving SINR `S/(N0+g(D))`. The per-user gap is

`log2((N0+S)(N0+g(D)) / (N0 (N0+g(D)+S)))  →  log2(1 + g(D)/N0)`  as `S/N0 → ∞`,

which equals the Proposition-1 bound with `Y = g(D)`: **Prop. 1 is attained in the high-SNR limit** and the information gap
`J(π*) − sup_{Π_k} J` equals `ε_r(k)` for `k < D`. (With non-exchangeable observations the masked agents can break the tie from absolute
position and the gap is 0 — the information gap is a property of what the observation reveals, not of the network.)

## 4. Mapping to the review's §4 points

| review point | status |
|---|---|
| 1 dimensions | every bound is in return units via `R̄`, `T`, `ε_r` |
| 2 `L(k)` circular | replaced by `ε_r(k)`, defined from `g, N0, geometry`; measured in `measure_Lk.py` |
| 3 wrong quantity (per-source vs aggregate) | Prop. 2: aggregate, exponent `2−α`, summable iff `α>2` |
| 4 unrealisability | `ε_info(k)` term in Prop. 3; not bounded by `ε_r` (stated as open) |
| 5 horizon | Prop. 3 `T(T+1)/2` for BC; `O(T)` for DAgger cited, not implemented |
| 6 tightness | §3: Prop. 1 attained as `S/N0→∞`; information gap equals `ε_r(k)` in the example |
