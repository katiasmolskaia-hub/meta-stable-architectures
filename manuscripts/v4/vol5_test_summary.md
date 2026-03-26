# Volume V Test Summary

This note keeps only the effective test conclusions for Volume V after the latest debugger pass.

## 1. What Volume V is trying to show

Adaptive RC is not a hard controller.
It is a bounded, state-aware coordinator that:
- reads group dispersion, lag, suppression, and anchors;
- adjusts gain gently instead of forcing uniform timing;
- keeps memory separate from temporary suppression;
- aims for recoverable diversity, not identical recovery times.

The correct claim is:

`bounded, recoverable coordination under heterogeneity`

not

`the fastest controller wins`.

## 2. Effective test families

### A. Toy adaptive RC sanity

Goal:
- verify that the adaptive layer runs without destabilizing the recovery loop.

Conclusion:
- adaptive RC is alive and bounded;
- the earliest toy run did not need to beat fixed RC immediately;
- the main result was that the adaptive layer could be introduced safely.

### B. Heterogeneous tail shaping

Goal:
- check whether adaptive RC changes the recovery tail, not only the mean.

Conclusion:
- on symmetric groups the effect is mostly on mean timing;
- on heterogeneous groups the recovery tail becomes visible;
- spread-friendly presets produce a longer but still bounded delayed tail;
- this is closer to metastability than to hard equalization.

### C. Infection-over-time profile

Goal:
- measure how many agents remain above the suppression threshold after stress.

Conclusion:
- most agents remain suppressed immediately after stress;
- release usually begins over the next 1-2 time units;
- spread-friendly settings preserve a controlled delayed tail rather than a collapse cascade.

### D. Multi-episode response

Goal:
- check whether the adaptive layer learns across episodes.

Conclusion:
- the controller is responsive to state;
- strong episode-to-episode learning is not yet cleanly demonstrated;
- this should be treated as an open problem, not as a solved claim.

## 3. Group memory `K_g`

### Intended role

`K_g` is supposed to be:
- bounded;
- episode-sensitive;
- separate from raw momentary suppression;
- a weak modifier of soft control, not a hidden permanent booster.

### What the debugger pass showed

The previous implementation had three distortions:
- phase dispersion was being used before the current-step value was computed;
- `K_g` had drifted toward a continuous accumulator instead of crisis-gated memory;
- recovery metrics mixed stress duration with post-stress recovery.

These issues made `K_g` look stronger and cleaner than it really was.

### What changed after the fix

After the fix:
- state variables are read in the correct order;
- `K_g` updates only when the crisis score crosses a threshold;
- recovery metrics are measured from stress end rather than stress start.

This produced a more honest picture:
- memory effects still exist;
- they are weaker than the old raw outputs suggested;
- they look more like event-triggered tail reduction than deep long-term learning.

## 4. Current network-level reading

### Light network run

After the fix and soft crisis gate:
- `with K_g` shows modest improvement over `no K_g`;
- post-stress recovery is a bit faster;
- tail spread and p90 are slightly smaller.

Interpretation:
- network memory is plausible and not empty;
- but the effect is modest.

### Repeated-stress runs

After the fix:
- `K_g` no longer acts like a constant invisible gain boost;
- repeated-stress runs show small tail compression;
- however, later episodes are still not dramatically better than early ones.

Interpretation:
- the model now supports weak event memory;
- it does not yet support a strong "the group learns across episodes" claim.

## 5. Safe claims for the manuscript

Volume V can safely claim that:
- fixed RC can be extended into a bounded state-aware coordinator;
- heterogeneous recovery tails can be shaped without turning control rigid;
- a bounded group memory layer is possible;
- network tests show weak but real adaptive effects.

Volume V should not yet claim that:
- strong multi-episode learning is already demonstrated;
- `K_g` robustly improves every network scenario;
- the memory layer is already theoretically clean at publication level.

## 6. Remaining open checks

The main unfinished task is now clear:

- decide whether Volume V is about `state-aware coordination` or about `memory across episodes`.

At the moment, the evidence is much stronger for the first than for the second.

That is not a failure.
It just means the paper should center the adaptive coordinator, while treating strong group-memory learning as an extension or future direction.
