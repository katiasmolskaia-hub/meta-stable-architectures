# Instructor Model v2 Research Plan

This is the practical execution order for Volume V after the conceptual reset.

## 1. Main rule

Do not introduce everything at once.

The order should be:
- observation first;
- routing second;
- memory third.

If memory is introduced too early, it hides whether the instructor actually sees the field.

## 2. Stage 0: lock the narrative

Goal:
- keep Volume V centered on the instructor-like adaptive coordinator.

Actions:
- use `Adaptive RC (Instructor Model)` as the stable name;
- treat `K_g` as secondary, not as the headline result;
- keep the main claim as bounded recoverability under heterogeneity.

Success condition:
- the manuscript no longer sounds like it is mainly about strong long-horizon learning.

## 3. Stage 1: add observation variables

Goal:
- make the coordinator state-reading, not only rhythm-giving.

Add to the model:
- `E_i` = local error or task mismatch;
- `A_i` = anchor reliability;
- improved `L_i` or `L(t)` = lag summary.

Minimal practical definitions:
- `E_i`: mismatch to target rhythm or target local behavior;
- `A_i`: high when suppression is low and error is low;
- `L_i`: delayed release or delayed return after stress.

Questions to test:
- does adding `E_i` separate confusion from suppression?
- does `A_i` identify useful local guides?

Success condition:
- the new observables vary meaningfully across agents during stress and recovery.

## 4. Stage 2: add focus routing

Goal:
- route influence through reliable neighbors instead of uniform coupling.

Add:
- anchor-weighted local influence;
- anti-contagion damping from unstable or high-error neighbors.

Compare:
- fixed RC;
- adaptive RC without anchors;
- Instructor v2 with anchors and anti-contagion.

Metrics:
- post-stress recovery time;
- p90;
- tail span;
- agent std;
- fraction above threshold at `t+1`, `t+2`, `t+4`.

Success condition:
- Instructor v2 gives either lower tail spread or safer delayed recovery without turning the group rigid.

## 5. Stage 3: spatial topology

Goal:
- move from abstract graph tests toward the class-like field you described.

Add topologies:
- local distance graph;
- local distance + small-world shortcuts;
- clustered layout graph.

Optional practical modifiers:
- hearing falloff with distance;
- visibility falloff with distance;
- local anchor amplification.

Questions to test:
- do local anchors help more in spatial networks than in abstract ones?
- does contagion spread locally before becoming global?

Success condition:
- spatial structure changes the outcome in a way that matches the intuition from real group behavior.

## 6. Stage 4: safety mode

Goal:
- represent the fact that risky tasks require sharper central commands.

Add:
- a scalar `U(t)` or `command_mode`.

Interpretation:
- low-risk state: soft broad guidance;
- high-risk state: sharper shorter signal.

Questions to test:
- does sharper guidance reduce contagion near difficult episodes?
- does it preserve bounded diversity instead of collapsing all timing?

Success condition:
- a risk-triggered mode helps in difficult windows without acting like hard control.

## 7. Stage 5: compressed group memory

Goal:
- reintroduce `K_g` only after the instructor already sees the field.

Rule:
- `K_g` must update from compressed episode summaries, not from raw agent accumulation.

Preferred structure:
- raw episode buffer;
- compressed episode score;
- gated update to `K_g`.

Suggested episode summary:
- crisis severity;
- post-stress recovery quality;
- tail span;
- dispersion;
- local contagion spread.

Questions to test:
- does `K_g` improve later responses once observation and routing are already present?
- does it stay bounded and event-triggered?

Success condition:
- memory gives modest but interpretable improvement without saturating into a constant gain bias.

## 8. Stage 6: co-adaptation

Goal:
- test whether instructor and group can improve together over repeated sessions.

Do not start here.
This is the last stage.

Interpretation:
- agents become less error-prone under repeated structured guidance;
- instructor becomes better at reading risk and routing attention.

Questions to test:
- do later sessions show better tail handling?
- do anchors emerge earlier?
- does contagion become more localized and easier to stop?

Success condition:
- there is clear improvement beyond simple state responsiveness.

## 9. Recommended experiment order

Run these in order:

1. Toy heterogeneity with `E_i` and `A_i`
2. Small network with anchor routing
3. Spatial network with visibility/hearing falloff
4. Safety mode on difficult stress windows
5. Repeated episodes without `K_g`
6. Repeated episodes with compressed `K_g`

This order keeps the interpretation clean.

## 10. What to write in the manuscript at each step

If only stages 1-2 work well:
- Volume V is about state-aware soft coordination.

If stages 3-4 also work:
- Volume V becomes a stronger instructor-field paper.

If stage 5 works:
- add bounded episode memory as secondary result.

If stage 6 works:
- only then talk about genuine group learning.

## 11. Immediate next coding task

The next coding target should be:

`add local error E_i and anchor reliability A_i to the network simulator, then compare fixed RC vs adaptive RC vs Instructor v2 on one heterogeneous spatial proxy.`

That is the cleanest next move for the project.
