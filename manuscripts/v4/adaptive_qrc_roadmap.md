# Adaptive RC Roadmap

This note is the working map for Volume V.
It is not the manuscript itself. Its purpose is to keep the idea, claims, and validation logic compact and consistent.

## 1. Core claim

Volume V extends Volume IV from a fixed RC to a state-aware RC.

The coordinator should:
- read the current group state rather than impose a fixed rhythm;
- stay soft and bounded rather than become a hard synchronizer;
- preserve recoverability under heterogeneity;
- allow structured recovery tails instead of collapsing everyone into the same timing;
- use group memory only after meaningful crisis episodes.

The main claim is:

`Adaptive RC is a bounded coordinator for heterogeneous groups, not an optimizer for the fastest possible recovery.`

## 2. What Volume V adds to Volume IV

Volume IV established:
- network-level recovery under a fixed RC;
- coherence-driven relaxation;
- bounded coordination that rescues the group from absorbing isolation.

Volume V should add only three new elements:
- state-aware gain modulation;
- anchor-sensitive / anti-contagion weighting;
- bounded group memory `K_g` that updates only after meaningful episodes.

Anything beyond that is optional and should not become the center of the paper.

## 3. Variables and roles

`S_i`
- temporary suppression / unavailability;
- current state, not memory.

`K`
- agent-level structural memory from earlier volumes.

`K_g`
- group-level memory of meaningful crisis episodes;
- must stay bounded;
- must not behave like a permanent gain offset.

`D(t)`
- phase dispersion;
- measures how broken the common rhythm is.

`L(t)`
- lag proxy;
- measures delayed recovery / delayed adaptation.

`A_i`
- anchor proxy for stable agents;
- used to increase attention to locally reliable nodes.

`C(t)` or `C_E`
- contagion or crisis score;
- used to decide when memory should update.

## 4. What the model should do

The coordinator loop is:

1. Observe dispersion, lag, suppression, and anchor structure.
2. Update bounded control terms such as `phi_gain` and `kappa`.
3. Reweight coupling toward stable agents and away from unstable ones.
4. Apply memory only if the episode is significant enough.

The intended behavior is:
- stronger help when the group is dispersed or delayed;
- weaker intervention when the group is already stable;
- no hidden conversion of memory into constant control pressure.

## 5. What to test

The evaluation stack should stay narrow.

### A. Sanity

- adaptive RC remains bounded;
- adaptive RC does not break recovery compared to fixed RC.

### B. Tail shaping

- compare fixed RC vs adaptive RC on heterogeneous groups;
- inspect mean recovery, p90, max, std, and tail span;
- confirm whether adaptive RC changes the shape of the recovery tail.

### C. Infection profile

- track fraction of agents above suppression threshold after stress ends;
- show whether delayed release is bounded rather than cascading.

### D. Memory

- compare `no K_g` vs `with K_g`;
- verify that memory updates only on meaningful episodes;
- verify that memory decays or relaxes rather than saturating permanently;
- show whether later episodes change in a measurable way.

## 6. What not to claim

Do not claim:
- universal learning across episodes;
- strong performance gains in all network settings;
- that `K_g` already produces deep long-term learning.

At the current stage, the evidence supports:
- bounded state-aware coordination;
- weak but real tail reduction in some network settings;
- event-triggered memory effects;
- no clear strong episode-to-episode learning yet.

## 7. Current status after debugger pass

The main implementation issues found in the recent pass were:
- phase dispersion was used before being computed for the current step;
- group memory had drifted toward a continuous accumulator instead of a crisis-gated memory;
- recovery metrics mixed stress duration with post-stress recovery.

After fixing those points:
- `K_g` behaves more honestly as an event-triggered memory;
- some network runs show modest tail reduction and slightly faster post-stress recovery;
- the long repeated-stress runs still do not show strong episode learning.

That is an acceptable Volume V result if the paper is framed correctly.

## 8. Recommended manuscript structure

1. Motivation: why fixed RC is not enough.
2. State-aware control law.
3. Anchor weighting and anti-contagion logic.
4. Group memory `K_g` and crisis gate.
5. Validation:
   - toy sanity;
   - heterogeneous tail shaping;
   - infection profile;
   - network memory checks.
6. Honest conclusion:
   - adaptive RC works as bounded soft coordination;
   - group memory is plausible but still weak;
   - stronger episode learning remains future work.
