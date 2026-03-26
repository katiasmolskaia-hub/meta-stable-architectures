# Instructor Model v2 for Volume V

This note defines the new working center for Volume V.
It is written as a practical concept note for the project, not as formal manuscript text.

## 1. Short answer on the name

You do not need to rename the whole project again.

The clean solution is:
- keep `RC` as the family name from Volume IV;
- keep `Adaptive RC` as the technical continuity term;
- use `Instructor Model` as the human-readable interpretation of the adaptive version.

So the safest wording is:

`Volume V: Adaptive RC as an Instructor Model`

This preserves continuity and avoids making it look like an unrelated theory.

## 2. New center of Volume V

The right center is not:
- "a smarter metronome";
- "a memory that stores everything";
- "a controller that makes everyone recover as fast as possible".

The right center is:

`a state-reading, error-aware, soft coordinator for heterogeneous groups`

In plain language:
- the coordinator does not only give rhythm;
- it observes how the group is responding;
- it notices lag, confusion, instability, and local anchors;
- it adjusts rhythm, emphasis, and coupling without forcing uniformity.

This is exactly closer to how a real human instructor works.

## 3. Human analogy translated into model terms

Real group exercise gives us a useful network interpretation.

The instructor:
- knows the intended movement pattern;
- keeps tempo and count;
- does not correct every individual at every second;
- watches the field;
- notices who lags, who is unstable, who can serve as a local example;
- reduces error contagion by redirecting attention;
- changes command sharpness when the exercise becomes risky.

This suggests that the coordinator should have four roles:
- rhythm source;
- state observer;
- error detector;
- local focus regulator.

That is stronger than the old "soft metronome" picture and more practical.

## 4. Core objects of Instructor Model v2

### 4.1 Global rhythm

Keep a global phase or rhythm variable:
- `Phi(t)`

Role:
- maintain shared temporal structure;
- provide the common count or beat.

This stays from Volume IV.

### 4.2 Task template

Add a simple reference pattern:
- `T`

Role:
- represent what the group is trying to do;
- define the expected direction, phase, or movement style.

At first this does not need to be complex.
It can be:
- target phase,
- target response window,
- or expected relation between local state and timing.

Why it matters:
- a human instructor does not only hold rhythm;
- the instructor also knows what "correct enough" looks like.

### 4.3 Agent state

Each agent should still have:
- local phase / motion state;
- suppression or temporary unavailability `S_i`;
- local memory / experience if needed.

Interpretation:
- `S_i` means "currently not available to follow well";
- not "has learned".

### 4.4 Group dispersion

Keep:
- `D(t)`

Role:
- measure how broken the shared rhythm is.

### 4.5 Lag field

Keep and improve:
- `L_i(t)` or group summary `L(t)`

Role:
- capture delayed response;
- identify who is consistently behind.

This is important because in real groups, some people are not unstable, just delayed.

### 4.6 Error field

Add:
- `E_i(t)`

Role:
- represent mismatch between current behavior and the target pattern;
- distinguish "off-beat / off-task" from simple suppression.

Examples:
- phase mismatch to target;
- local motion mismatch to neighborhood pattern;
- delayed or incorrect execution of the task.

This is one of the most important additions.

### 4.7 Anchor reliability

Add:
- `A_i(t)`

Role:
- identify agents who are currently reliable enough to serve as local guides.

This should not mean "best forever".
It should mean:
- visible,
- stable,
- low suppression,
- low error,
- reasonably synchronized.

### 4.8 Contagion risk

Add:
- `C_i(t)` or local edge-level contagion weights

Role:
- estimate how likely an unstable node is to destabilize neighbors.

This directly matches the real observation:
- one confused person can disturb nearby people;
- not everyone needs to see that confusion.

### 4.9 Group memory

Keep:
- `K_g`

But redefine its role clearly:
- not full group history;
- not direct integration of all agent signals;
- only a compressed memory of meaningful episodes.

Best interpretation:
- `K_g` is the compressed trace of what kind of crisis happened and how the group survived it.

## 5. Control law of Instructor Model v2

The coordinator should change three things.

### 5.1 Rhythm intensity

`phi_gain`

Increase when:
- dispersion is high;
- lag grows;
- error field grows;
- contagion risk grows.

Decrease when:
- the group is stable enough.

### 5.2 Focus routing

The coordinator should not listen equally to every node.

Influence should shift:
- toward reliable anchors;
- away from unstable or misleading nodes.

That means local coupling weights should depend on:
- `A_i`,
- `S_i`,
- `E_i`,
- contagion risk.

### 5.3 Command sharpness / safety mode

Add a simple scalar mode:
- `U(t)` or `command_mode`

Role:
- represent how concise and strong the instructor signal becomes when the situation is risky.

Interpretation:
- in easy states the signal is soft;
- in risky states the signal becomes shorter, clearer, more central.

This is directly motivated by your real teaching practice.

## 6. Minimal mathematical structure

The model does not need a huge formal jump at first.

You can define the instructor gain as

```text
phi_gain(t) = clamp(phi0 * [1 + aD*D(t) + aL*Lbar(t) + aE*Ebar(t) + aC*Cbar(t) + aK*K_g(t)])
```

Anchor weight can be built from low suppression and low error:

```text
A_i(t) ~ (1 - S_i) * (1 - E_i)
```

Coupling can then prefer reliable neighbors:

```text
w_ij ~ adjacency_ij * A_j * (1 - C_j)
```

And the instructor input can be sharper in risky moments:

```text
U(t) = soft baseline + emergency emphasis when risk is high
```

This is enough for a first practical model.

## 7. Topology: what you noticed in real groups

Your observation about self-organization by position is important.

That suggests the network should not only use abstract graph families.
It should also test:
- distance-based local visibility;
- hearing / attention falloff with distance;
- local clusters;
- small-world rewiring on top of local placement.

So instead of only:
- ring,
- Erdos-Renyi,
- scale-free,

we should add:
- spatial local graph;
- spatial small-world graph;
- clustered class-layout graph.

This is much closer to real field behavior.

## 8. What to test first

Do not test everything at once.

### Stage 1. Error-aware instructor without learning

Goal:
- test whether state-reading plus anchor routing helps more than fixed RC.

Add:
- `E_i`,
- `A_i`,
- contagion-aware reweighting.

Compare:
- fixed RC;
- adaptive RC;
- Instructor v2.

Main metrics:
- post-stress recovery time;
- tail span;
- p90;
- fraction suppressed at t+1, t+2, t+4;
- contagion spread around destabilized nodes.

### Stage 2. Spatial topology

Goal:
- test whether local visibility / hearing structure changes the outcome.

Compare:
- ring;
- spatial local graph;
- spatial small-world graph.

Main question:
- do local anchors help more in spatially grounded networks?

### Stage 3. Safety mode

Goal:
- test whether sharper central guidance helps in risky phases.

Main question:
- does emergency emphasis reduce contagion without collapsing diversity?

### Stage 4. Group memory

Goal:
- add compressed episode memory only after the state-reading layer works.

Main question:
- does `K_g` improve later episodes once the instructor already sees error and anchors?

This order matters.
Memory should come after observation, not before.

## 9. What not to do

Do not:
- make `K_g` the main intelligence of the system;
- store all agents in memory;
- introduce too many new variables into the manuscript at once;
- present "tail shaping" and "learning" as the same thing;
- rename the project so much that continuity gets lost.

## 10. Recommended Volume V structure now

### Main thesis

Adaptive RC becomes an Instructor Model when it does three things:
- reads the state of the group;
- detects local error and anchors;
- routes attention and influence to stabilize the field softly.

### Secondary thesis

Group memory `K_g` stores compressed traces of meaningful crisis episodes and can weakly bias later coordination.

### Honest result path

First show:
- adaptive state-reading helps.

Then test:
- anchor and anti-contagion logic.

Only then claim:
- bounded event memory can shape later response.

## 11. Immediate implementation plan

1. Keep the title line conceptually stable:
   - `Adaptive RC (Instructor Model)`
2. Add a new note or section in the manuscript explaining:
   - instructor as state-reading coordinator
3. Introduce one new observable first:
   - `E_i` as local error / task mismatch
4. Introduce one new routing signal:
   - `A_i` as local anchor reliability
5. Build one spatial network experiment:
   - local distance-based graph with uneven visibility
6. Compare:
   - fixed RC vs adaptive RC vs Instructor v2
7. Only after that reintroduce `K_g` on top.

## 12. Bottom line

The new correct center for Volume V is:

`not a better metronome, but a soft instructor that reads the field`

That keeps continuity with Volume IV, uses your real teaching insight correctly, and gives the project a more practical and believable direction.
