# Performance Gap Bound Theorem for k-Masked MARL

## 1. Problem Formulation
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ be a Markov Game for $N$ UAVs.
- The **global state** $s \in \mathcal{S}$ represents the exact positions of all UAVs and ground users. (Full-CSI)
- The **joint action** $\mathbf{a} = (a_1, ..., a_N) \in \mathcal{A}$ comprises continuous trajectories and discrete channel assignments.

Let $\pi^*$ be the optimal joint policy derived from the centralised, full-CSI solver (e.g., MINLP). 
Let $V^*(s)$ be the optimal value function under full observability.

## 2. Observability Constraint
During decentralized execution, UAV $i$ only receives a local observation $o_i^k = M_k(s, i)$, where $M_k$ is a masking function that filters out any entity at a distance greater than $k$ from UAV $i$.
Let $\hat{\pi}_k$ be the $k$-masked neural policy obtained via asymmetric distillation (behavioral cloning).

## 3. The Performance Gap Theorem

**Theorem 1 (Suboptimality of Range-Limited Distillation):**
Let $D_{TV}(\pi^*(\cdot|s) || \hat{\pi}_k(\cdot|o_i^k))$ be the total variation divergence between the full-CSI expert policy and the decentralized $k$-masked policy. 

Assume the environment's transition dynamics $P$ and reward function $R$ are Lipschitz continuous with respect to the state components. 
Let $\epsilon_{approx}$ be the fundamental function approximation error of the neural network.
Let $L(k)$ be the **information loss function**, defined as the maximum expected impact on the optimal action choice caused by state variables outside the radius $k$.

For any state $s$, the value function gap is bounded by:

$$ | V^*(s) - V^{\hat{\pi}_k}(s) | \leq \frac{2 \gamma}{(1-\gamma)^2} \sup_s \left( \epsilon_{approx} + L(k) \right) $$

### 3.1. Properties of $L(k)$
- **Monotonicity:** $L(k)$ is monotonically decreasing with respect to $k$. As $k \to \infty$ (global visibility), $L(k) \to 0$.
- **Network Density Dependency:** In dense user scenarios, the interference from distant UAVs (beyond $k$) decays inversely with distance, meaning $L(k) \sim \mathcal{O}(1/k^\alpha)$ where $\alpha$ is the path-loss exponent.

## 4. Phase 3: Closing the Residual Gap
The pre-training (Phase 2) minimizes $\epsilon_{approx}$ but cannot overcome the theoretical floor set by $L(k)$. 

To surpass this, we employ **Online Trust-Region Fine-Tuning** (e.g., MAPPO/MATR) with a **Range-Masked Critic**. 
The critic learns a localized value function $\hat{V}_k(o_i^k)$ that accurately evaluates the expected return *given the limited information*, rather than attempting to estimate the full-CSI $V^*(s)$. This aligns the actor and critic representations, allowing the policy to adapt its trajectory to actively compensate for the unknown state space beyond $k$ (e.g., by learning conservative spacing).
