# Introduction and Literature review (prose, paste-ready) + document fix list

Companion to `claude/v5_paper_outline.md` and `claude/v5_methods.md`. Part A and B are prose to paste into
`Observability_paper.docx`. Part C is the edit list for the rest of that document.

**On citations.** Every reference below is a marked slot of the form `[CITE-n: what this must be]`. None are
invented. Fill them from the project's existing paper list; the slot text says what each one has to establish so a
wrong substitution is obvious.

---

# Part A -- Introduction (paste under "Introduction", replacing the five bullets)

Unmanned aerial vehicles acting as mobile relays can raise the spectral efficiency of a ground network by moving
toward demand, but a swarm sharing a small number of channels faces two coupled decisions at every step: where each
vehicle should fly, and which channel it should use. The coupling is what makes the problem hard. Position determines
the useful signal for the users a vehicle serves; the channel determines which other vehicles contribute interference
to those same users; and because interference adds over all co-channel transmitters, a decision taken by one vehicle
changes the value of every other vehicle's decision. [CITE-1: survey or canonical treatment of UAV-assisted
communications and the placement/trajectory problem.]

Two families of methods address this, and each pays a different price. Classical trajectory optimisation --
successive convex approximation, block-coordinate descent, mixed-integer formulations -- produces strong solutions but
needs seconds to hours per scenario, which rules it out for closed-loop flight, and it needs a centralised view of the
whole system. [CITE-2: representative classical UAV trajectory/channel optimisation.] Multi-agent reinforcement
learning produces a decentralised policy that runs in milliseconds, but conventionally starts from random weights and
consumes millions of environment interactions before it is useful. [CITE-3: representative MARL for UAV swarm control.]
The obvious reconciliation is to distil the optimiser into a neural policy: solve offline where time is cheap, execute
online where it is not. That is the pipeline we build, and it works -- but in the course of building it we found that it
answers a less interesting question than the one that actually governs performance.

**The question that governs performance is what each agent is allowed to see.** A decentralised relay cannot observe
the whole system; it observes users and peers within some radius, and that radius is usually chosen by convention
rather than measured. We find that this choice dominates everything else we varied. On a frozen test set of 1024
scenarios, a policy restricted to a six-unit observation radius attains 0.353 of the range between a do-nothing
baseline and the planner, while the same architecture with unrestricted observation attains 0.799. Nothing else in our
study -- not the training method, not the optimiser's hyperparameters, not the critic's information -- moves
performance by a comparable amount. Crucially, the two halves of the observation are not equally valuable: widening
the *user* set recovers 2.0% of that gap, while widening the *peer* set recovers 98.9%. The binding constraint is
knowing what the other vehicles are doing, not where the users are.

This reframes the engineering problem. If peer information is what matters, the question becomes how little of it
suffices, and in what form it can actually be obtained in flight. We study two signals that a deployed system could
plausibly provide. The first is a per-channel count of other vehicles, broadcast at a cost of C numbers per vehicle per
step. The second requires no communication at all: the co-channel received power measured at the vehicle's own
position, which is what a receiver co-located with it would read, and which approximates what its own served users
experience. Individually these recover 80.9% and 47.8% of the gap. Jointly they recover 93.3% -- and a
radius-limited policy holding both outperforms a fully observing policy trained from scratch.

Distillation earns its place in this story as the instrument that made the study affordable rather than as the
headline. Each configuration costs roughly eighteen minutes of offline work per replicate, against hours of online
reinforcement learning, which is what allowed us to run the ablation at all. We report its sample efficiency honestly:
at a matched small online budget the distilled start is ahead by 24.95 returns, and it reaches a given performance
level with two-to-four-fold fewer online samples; but at convergence a from-scratch agent matches it. The claim is
speed, not a better final policy.

Finally, the online fine-tuning step that is supposed to close the remaining gap does something the aggregate return
hides. It raises mean throughput by 15.3, and it does so partly by abandoning the worst-served users: the tenth
percentile user rate falls by 33% and outage nearly doubles, on every replicate we examined. This is not a failure of
the algorithm but a faithful response to the objective, which is utilitarian and places no floor on any individual
user. We show that the degradation is separable from the gain -- across replicates, the size of the coverage loss is
uncorrelated with the size of the return improvement -- and that adding a coverage floor to the training reward
removes it: the constrained policy retains the throughput gain while ending with *better* coverage than the policy it
was initialised from, and it gets there by reducing co-channel crowding rather than by redistributing service.

**Contributions.**
1. A measurement of information sufficiency for decentralised UAV relay control, separating peer visibility from user
   visibility, on a frozen test set with a stated normalisation (Section [RESULTS-1]).
2. Two deployable observation features -- a per-channel broadcast count and a receiver-side interference measurement --
   with matched-dimension placebo and radius-restricted controls, shown to be complementary and to recover 93.3% of the
   observability gap (Section [RESULTS-2]).
3. A range-bounded asymmetric distillation pipeline, with an honest accounting of its sample efficiency in team steps
   including the offline stage (Section [RESULTS-3]).
4. A diagnosis of what online fine-tuning changes, and a single reward term that removes its coverage cost at no
   measurable cost in return (Section [RESULTS-4]).
5. An evaluation protocol for this class of study, including the finding that the heuristic planner commonly used as a
   normaliser does not converge with compute, so normalised scores are a convention that must name their reference
   (Section [METHODS-M11], Section [RESULTS-PROTOCOL]).

---

# Part B -- Literature review (paste under "Literature review")

**Classical trajectory and channel optimisation for UAV relays.** The placement and trajectory problem for aerial
relays is usually posed as maximising a sum-rate or coverage utility subject to speed, energy and altitude
constraints, and solved by successive convex approximation, block-coordinate descent, or mixed-integer programming
where the channel assignment is explicit. [CITE-4: SCA/BCD trajectory optimisation.] [CITE-5: mixed-integer or
convex formulation with channel/subcarrier assignment.] These methods give strong solutions and, in restricted
settings, optimality guarantees, but their runtime scales poorly with the number of vehicles and the horizon, and they
assume a centralised view. Our planner belongs to this family -- alternating gradient-based trajectory refinement with
exact channel best response -- and we use it as a teacher rather than as a controller. We are explicit that it carries
no optimality certificate: Section [METHODS-M8] reports that its objective continues to improve log-linearly with
compute budget and had not converged at ninety-six times our reference setting, which is a property we believe is
shared by many heuristics used as baselines in this literature and is rarely measured.

**Multi-agent reinforcement learning for aerial networks.** Decentralised control of UAV swarms has been approached
with independent learners, value factorisation, and centralised-training/decentralised-execution actor-critic methods,
of which MAPPO is the common contemporary choice. [CITE-6: MAPPO or CTDE method paper.] [CITE-7: MARL applied to
UAV swarm communication/coverage.] Work in this area typically trains from random initialisation and reports returns
after millions of environment steps. Our study positions against that convention in two ways: we quantify what the
distillation shortcut is actually worth in matched sample budgets, and we show that for a radius-limited agent the
achievable level is set by the observation rather than by the training budget, which bounds what any amount of online
training can deliver.

**Partial observability and what agents are given to see.** Decentralised control under partial observation is
formalised as a Dec-POMDP, and a large body of work studies architectures that consume variable-sized neighbour sets
-- graph networks, attention, and permutation-invariant pooling. [CITE-8: Dec-POMDP formalism.] [CITE-9: attention or
graph-based architecture for variable neighbour sets in multi-agent control.] The observation radius itself, however,
is usually a fixed hyperparameter justified by communication range rather than an object of study, and ablations over
it are rare. The closest related question in the MARL literature is asymmetry between actor and critic information,
where a centralised critic is given more than the actor. We include that ablation and report it as a bounded null
(Section [RESULTS-NULL]); our finding is that the actor's observation, not the critic's, is what matters here.

**Imitation and distillation from optimisers.** Learning a fast policy that reproduces a slow optimiser is studied as
amortised optimisation, learning-to-optimise, and policy distillation; in the sequential setting, naive behaviour
cloning suffers from compounding distribution shift, which DAgger addresses by querying the expert at states the
learner actually visits. [CITE-10: DAgger.] [CITE-11: amortised optimisation / learning-to-optimise for a
communications or control problem.] Our pipeline uses a receding-horizon re-solve at every visited state, which makes
the expert queryable off its own distribution -- a requirement that a one-shot open-loop dataset cannot satisfy.
Offline-to-online reinforcement learning studies the complementary question of fine-tuning a pre-trained policy with
environment interaction without destroying it. [CITE-12: offline-to-online RL fine-tuning.] Our online phase is a
plain MAPPO fine-tune, and the pathology we document -- the policy improving the scalar objective by sacrificing a
subpopulation -- is, to our knowledge, not reported in that setting, though it is closely related to a long-standing
concern in the resource-allocation literature, discussed next.

**Fairness and coverage in resource allocation.** That maximising a sum utility can starve the worst-served users is a
classical result in wireless scheduling, and the standard remedies are to change the utility -- proportional fair,
max-min fair, or alpha-fair objectives -- or to impose per-user rate constraints. [CITE-13: proportional-fair or
alpha-fair utility in wireless resource allocation.] [CITE-14: max-min fairness or per-user QoS constraints in UAV
placement.] Our contribution here is not the observation that utilitarian objectives are unfair, which is known, but
the measurement that an *online fine-tuning step applied to an already-fair distilled policy* re-introduces the
unfairness, that the unfairness is statistically separable from the performance gain rather than its price, and that a
floor term placed above the reporting threshold removes it while the policy finds a genuinely less-crowded channel
configuration.

**Empirical methodology in reinforcement learning.** Reported differences between deep RL methods are frequently
smaller than their run-to-run variance, and the community has converged on guidance about seed counts, paired
evaluation, and interval estimates. [CITE-15: reproducibility/statistical-practice paper in deep RL.] We adopt and
extend that guidance for the distillation setting, where the independent replicate is the offline student rather than
the online seed, and where the normalising reference is itself a heuristic whose quality depends on the compute spent
on it (Section [METHODS-M11]).

---

# Part C -- Document fix list for `Observability_paper.docx`

Keyed to searchable text, not paragraph numbers. Priority 1 blocks a readable draft; 3 is cosmetic.

## C1. Delete outright -- outline scaffolding visible to readers (Priority 1)
| search for | action |
|---|---|
| `Claims to state, in the order the evidence supports them:` | delete the line **and** the five bullets under it; Part A replaces them |
| `Companion to claude/v5_paper_outline.md (S2 and S3)` | delete the whole paragraph |
| `Points to make:` | delete the line; keep the bullets, convert to prose |
| `State clearly that this is not HARL, and why that is a threat (S9)` | delete this clause; the HARL point belongs in Threats, phrased as a limitation |
| `(write this before a reviewer does)` | delete from the "Threats to validity" heading |
| `-- worth one sentence in the paper` | delete the clause (end of the channel-switching paragraph) |
| `This belongs in the paper as a caution about probe-driven feature selection.` | delete; the preceding sentences already make the point |
| `We did not act on this -- say so, and state the resulting bound on the null (S8).` | delete; replace with a plain sentence stating the bound |
| `(review Q3 -- main contribution)`, `(review Q4 -- best new finding)` | delete from all headings |
| `-- Table T6`, `-- Table T5, Figure F4`, `-- Figure F5`, `-- Table T2, Figure F1`, `-- Table T4, Figure F3`, `-- Appendix B, Figure F7`, `-- tie into S7`, `(appendix)` | delete from all headings; these were my cross-references |

## C2. Rewrite in place -- imperative notes that should be statements (Priority 1)
| search for | rewrite as |
|---|---|
| `Note explicitly in the paper: $r$ has no floor on the worst-served user.` | "The reward places no floor on any individual user's rate; Section [RESULTS-4] shows the consequence." |
| `State this prominently. The planner is a strong heuristic, not a certified optimum` | "The planner is a strong heuristic and carries no optimality certificate; Section [METHODS-M11] shows it does not converge with budget." |
| `-- stated in the paper as a design recommendation we did not have the budget to adopt` | "We report this as a design recommendation for future studies; our own results use three students per configuration." |
| `This corrects the reviewer's "2.5-3x more channel agility" reading` | "A pooled switch rate conceals this: a policy that re-assigns once at t=1 already produces 0.033 switches per agent-step at C=3, T=20." |

## C3. Move to a Reproducibility appendix (Priority 2)
All six `Source: results2/...` paragraphs, plus the inline file paths in the channel-switching, variance-decomposition,
planner-ladder and probe paragraphs, and `regenerate with planner_check.py --n 64 --ladder`.
Create an appendix "Reproducibility: result files and regeneration commands" and move them there as a table of
*claim -> file -> command*. Do not delete them; a reviewer asking "where does 0.353 come from" is answered by this
appendix, and the regeneration commands are already listed at the end of `v5_paper_outline.md`.

## C4. Fix the broken mathematics (Priority 1)
- **The MAPPO loss is raw LaTeX split across two paragraphs.** Search `\mathcal{L}^{\mathrm{actor}}`. The second
  paragraph begins `c_{\mathrm{ent}}\mathbb{E}[\mathcal{H}]`. Rebuild as one display equation:
  the clipped surrogate minus `c_ent` times the entropy, with `rho` the probability ratio. Source text is M10 in
  `claude/v5_methods.md`.
- **Stray inline `$...$`.** Search the document for `$`. Eight paragraphs carry unconverted symbols: `$i$`, `$u$`,
  `$H$` (M2 opening), `$g$`, `$r$` (M3 closing), `$z_0$` (M7 opening), `$k$` (critic-mask paragraph), `$t$`
  (unit-of-analysis paragraph), and the bracketed hyperparameter list after the MAPPO loss. Convert each to a Word
  equation or to italic text.
- **Check `M4. Observation: the -masked set observation`** -- the heading lost its `k`. Should read
  "M4. Observation: the k-masked set observation".

## C5. Remove duplication (Priority 2)
- The short Methods summary (`Environment: N=8 UAVs...`, `Teacher: batched...`, `Student: set-based actor...`,
  `Distillation: open-loop BC...`, `Online: in-repo MAPPO...`) duplicates M1-M11 immediately below it. **Delete the
  five summary lines**; M1-M11 is the methods section.
- The `Evaluation protocols` section (3.1 seed sets, 3.2 planner ladder, 3.3 replicates) duplicates M11 and M8.
  **Fold its two tables into M11 and M8 and delete the section**, or keep it and cut the corresponding prose from M11
  -- but not both.

## C6. Renumber and restyle (Priority 3)
- Result sections currently run unnumbered, then `5.`, `6.`, `7.`, `8.`, `9.` -- inherited from my outline's own
  numbering. Renumber consecutively from whatever Results becomes (likely 4).
- `Abstract` is Heading 4; make it Heading 1 or an unnumbered style, consistent with the rest.
- Table captions (`Table T1 -- information ablation, TEST set (n=1024, strong++)`) are Heading 3 and will appear in the
  table of contents. Convert to a caption style and renumber Table 1..9 (drop the internal `T` labels).
- Internal tags `A7`, `A8_kinf`, `B_uinf`, `B_ainf`, `C_b1`, `C_b2`, `D_sense3..6`, `bcast=N` are run names, not paper
  names. Replace with descriptive labels throughout, e.g. `k=6 (baseline)`, `all users`, `all peers`,
  `full observation`, `broadcast count`, `sensing`, `broadcast+sensing`, `sensing (radius-limited)`,
  `sensing (shuffled placebo)`. Keep a mapping table in the reproducibility appendix.

## C7. Still empty, needs writing (Priority 1-2)
| section | status |
|---|---|
| Keywords | empty placeholder |
| Literature review | **Part B above** |
| Discussion | empty -- draft after the figures exist |
| Conclusion | empty |
| References | empty -- fill the `[CITE-n]` slots |
| Results 2, 3, 4 headings | headings present, body empty; the material exists in `v5_paper_outline.md` S5-S7 |
