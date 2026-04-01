# Volume VI Route Diagnostics Working File

## Why this file exists

This file is meant to prevent the early Volume VI experiment line from scattering across too many small notes.

Its role is simple:

- hold the strongest current diagnostic results in one place,
- preserve the current working plateau,
- and make it easier to restart from the same point in a new chat or a later session.

This is not a polished manuscript section.

It is a working synthesis file.

## Current experimental base

The current route experiment lives in:

- `simulations/experiment_volume_vi_bond_routes.py`

The current output files live in:

- `outputs/volume_vi_bond_routes/summary.csv`
- `outputs/volume_vi_bond_routes/aggregate_summary.csv`
- `outputs/volume_vi_bond_routes/routes_overview.png`
- `outputs/volume_vi_bond_routes/drift_state_sequence.csv`
- `outputs/volume_vi_bond_routes/shock_state_sequence.csv`
- `outputs/volume_vi_bond_routes/distortion_state_sequence.csv`

The experiment currently compares three route types:

- `drift route (маршрут дрейфа)`,
- `shock route (маршрут удара)`,
- `distortion route (маршрут искажения)`.

## Three strongest current indicators

At the present stage, the three most effective diagnostic indicators are:

### 1. Tail-sensitive diagnostics

The routes are not best distinguished only by final state.

They are strongly distinguished by the shape of the recovery tail.

The most useful measures so far are:

- `tail area (площадь хвоста)`,
- `recovery time to threshold (время возврата к порогу)`,
- `re-entry tail area (площадь хвоста повторного входа)`,
- `time in restored-weak-only regime (время только в режиме восстановленных слабых связей)`.

Current aggregate pattern:

- `shock (удар)` has a short and small tail,
- `drift (дрейф)` has a longer weakened tail,
- `distortion (искажение)` has a very large non-recoverable tail.

### 2. State-transition sequence

The routes are also distinguishable by the order of dominant bond-state changes.

Current dominant reading:

- `shock (удар)` passes through `crisis-support bond (кризисно-поддерживающую связь)` and returns to `stabilizing bond (стабилизирующей связи)`,
- `drift (дрейф)` passes through crisis support but falls back into weaker post-crisis states before returning only to `restored weak bond (восстановленной слабой связи)`,
- `distortion (искажение)` continues onward into `distorted bond (искаженную связь)`.

This is important because route diagnosis is now visible not only in how the system ends, but in how it passes through bond states.

### 3. State occupancy

The routes differ not only by transition order, but by how long they remain in different bond regimes.

The most useful measures here are:

- `dwell time in stabilizing state (время в стабилизирующем состоянии)`,
- `dwell time in restored-weak state (время в восстановленно-слабом состоянии)`,
- `dwell time in distorted state (время в искаженном состоянии)`,
- and `number of transitions (число переходов)`.

Current aggregate pattern:

- `shock (удар)` spends most of its post-crisis time in `stabilizing (стабилизирующем)` mode,
- `drift (дрейф)` spends much more time in `restored weak (восстановленно-слабом)` mode,
- `distortion (искажение)` spends a large final block in `distorted (искаженном)` mode.

## Current aggregate reading

The present repeated-run pattern suggests the following:

- `shock route (маршрут удара)` is an acute but recoverable route,
- `drift route (маршрут дрейфа)` is a prolonged weakening route with a long recovery tail,
- `distortion route (маршрут искажения)` is a qualitatively pathological route with collapse into distorted bonds.

This distinction now appears in:

- final trajectory shape,
- tail diagnostics,
- state-transition order,
- and state dwell times.

## What already looks robust

Across the current multi-seed sweep, the following signatures remain stable:

- short tail and strong stabilizing dwell for `shock (удара)`,
- long restored-weak tail for `drift (дрейфа)`,
- large distorted dwell and failed re-entry for `distortion (искажения)`.

So the current route differences no longer look like one-off numerical accidents.

## Wider sweep result

The route diagnostics line has now also been tested under a wider sweep that includes:

- more random seeds,
- softer stress variation,
- harder stress variation,
- more adaptive dynamics,
- and less adaptive dynamics.

The widened sweep preserves the main route ordering:

- `shock route (маршрут удара)` remains the most recoverable,
- `drift route (маршрут дрейфа)` remains slower and more tail-heavy,
- `distortion route (маршрут искажения)` remains pathological.

This is important because the distinction now survives not only random initialization, but also modest changes in model regime.

### Strongest new result from the wider sweep

The most informative new result appears under `stress_hard (жестком стрессе)`.

Under that regime:

- `drift (дрейф)` begins to collapse into a distorted final state,
- while `shock (удар)` still remains recoverable.

This suggests that the routes are not only different in mild settings.

They may also have different `failure boundaries (границы срыва)`.

### Working interpretation

At the current stage, the route diagnostics line suggests three qualitatively distinct behaviors:

- `shock (удар)` = acute but re-stabilizing crisis,
- `drift (дрейф)` = prolonged weakening with a long post-crisis tail,
- `distortion (искажение)` = harmful relational reorganization with failed re-entry.

Under stronger stress, `drift (дрейф)` may cross into a failure regime before `shock (удар)` does.

That boundary asymmetry is one of the strongest current signs that the route distinction is structurally meaningful.

## Partial bond heterogeneity result

The next strengthening step introduced `partial bond heterogeneity (частичную неоднородность связей)`.

This means that the experiment no longer treats all bonds as one uniform mass.

Instead, the current prototype distinguishes at least:

- `local bonds (локальные связи)`,
- `bridge bonds (мостовые связи)`.

These bond types begin with different tendencies:

- local bonds are more stable and less free,
- bridge bonds are more flexible but more vulnerable.

The route pressure then acts on them differently.

### What changed after this strengthening

The earlier diagnostic picture did not disappear.

But it became sharper.

- `shock (удар)` remained strongly recoverable,
- `distortion (искажение)` remained fully pathological,
- and `drift (дрейф)` became more structurally interesting.

The key new result is that `drift route (маршрут дрейфа)` no longer behaves only like a long weakened recovery.

It now behaves more like a `mixed structural degradation (смешанная структурная деградация)`:

- part of the bond architecture remains viable,
- while another part begins to fall into distorted final states.

This suggests that drift may not simply weaken the field uniformly.

It may selectively damage the more vulnerable bond classes first.

### Why this matters

This strengthening makes the route diagnostics line more realistic.

The model now begins to suggest not only:

- which crisis route is present,

but also:

- which parts of the bond architecture are at greatest risk under that route.

That is an important step toward route-sensitive bond balancing rather than route-blind recovery.

### Class-level diagnostic reading

The current class-level diagnostics now suggest something stronger than a generic route distinction.

They suggest that different crisis routes threaten different parts of the bond architecture.

In the current prototype:

- `shock route (маршрут удара)` preserves both `local bonds (локальные связи)` and `bridge bonds (мостовые связи)` at a high level,
- `drift route (маршрут дрейфа)` preserves `local bonds (локальные связи)` much better than `bridge bonds (мостовые связи)`,
- `distortion route (маршрут искажения)` collapses both classes into a pathological regime.

The most important current signal is therefore:

- `drift (дрейф)` looks like a selective loss of bridge structure before it becomes a total collapse.

This is highly relevant for Volume VI because it means that route diagnosis may eventually guide not only general intervention, but bond-class-specific intervention.

In practical conceptual terms:

- `shock (удар)` calls for rapid temporary support and re-stabilization,
- `drift (дрейф)` calls for preservation or restoration of vulnerable bridge bonds,
- `distortion (искажение)` calls for limiting harmful bond reorganization itself.

### Two new working indices

The current prototype now also supports two compact diagnostic indices:

- `bridge loss index (индекс потери мостовых связей)`,
- `local preservation index (индекс сохранения локальных связей)`.

These indices are useful because they compress a large amount of route-specific information into a more readable diagnostic language.

Current working interpretation:

- high `bridge loss index (индекс потери мостовых связей)` with still-positive `local preservation index (индексом сохранения локальных связей)` suggests a `drift-like (дрейфоподобную)` crisis,
- low `bridge loss index (индекс потери мостовых связей)` with high `local preservation index (индексом сохранения локальных связей)` suggests a `shock-like (удароподобную)` crisis,
- high `bridge loss index (индекс потери мостовых связей)` together with collapsed `local preservation index (индексом сохранения локальных связей)` suggests a `distortion-like (искаженно-патологическую)` route.

This matters because it moves the model closer to a route-sensitive bond policy:

- not only identifying crisis type,
- but indicating which part of the bond architecture needs protection first.

## First intervention result

The first minimal `route-sensitive bond intervention layer (маршрутно-чувствительный слой помощи связям)` has now been tested.

The main result is already meaningful:

- `drift route (маршрут дрейфа)` improves strongly under bridge-preserving intervention,
- while `shock route (маршрут удара)` remains almost unchanged,
- and `distortion route (маршрут искажения)` remains difficult.

### What improved

In the current base regime:

- drift without intervention ends in a mixed degraded state with partial loss of bridge structure,
- drift with intervention returns to a much more recoverable condition.

This improvement appears in:

- higher `gatherability (собираемости)`,
- smaller `tail area (площади хвоста)`,
- and disappearance of final distorted bonds in the drift case.

### Why this matters

This is important because the intervention is not improving everything equally.

It is helping the route for which it was conceptually designed.

That means the route-sensitive intervention line is no longer only a theoretical idea.

It has now produced a first selective improvement result.

### What did not improve enough

The same first intervention layer does not yet meaningfully rescue:

- `distortion route (маршрут искажения)`,
- or `drift route (маршрут дрейфа)` under the harder stress regime.

This is also a useful result.

It means the current intervention is:

- strong enough for baseline bridge-preservation,
- but not yet strong enough for harder pathological drift or full distortion blocking.

### Current interpretation

The working intervention picture is now:

- `shock (удар)` mostly needs no extra rescue at the current model stage,
- `drift (дрейф)` can be significantly improved by protecting bridge structure,
- `distortion (искажение)` still requires a stronger dedicated blocking logic.

## Strengthened distortion-blocking result

The next refinement step strengthened `distortion blocking (блокировку искажения)` and tested it in a focused base comparison.

This produced a meaningful improvement for the pathological route.

### Before strengthening

Without intervention, the current `distortion route (маршрут искажения)` produced:

- failed `gatherability (собираемость)`,
- very large `tail area (площадь хвоста)`,
- and fully distorted final bond structure.

### After strengthening

With the stronger `distortion blocking (блокировкой искажения)`, the distortion route no longer remained in total collapse.

It moved into a partially recoverable regime:

- `gatherability (собираемость)` rose substantially above zero,
- `tail area (площадь хвоста)` dropped significantly,
- and final distorted bonds were reduced rather than remaining total.

### Why this matters

This is the first sign that the intervention layer is becoming genuinely route-sensitive in practice.

At the current stage:

- `bridge preservation (сохранение мостовых связей)` helps `drift (дрейф)`,
- `distortion blocking (блокировка искажения)` helps `distortion (искажение)`,
- and `shock (удар)` remains comparatively self-recoverable in the present model.

This means the project now has early evidence not only for route diagnosis, but for route-specific relational help.

## Current route-to-help map

The present working map can now be stated in one compact form:

- `shock route (маршрут удара)` -> `acute support scaffold (острый поддерживающий каркас)`
- `drift route (маршрут дрейфа)` -> `bridge preservation / bridge repair (сохранение и восстановление мостовых связей)`
- `distortion route (маршрут искажения)` -> `distortion blocking (блокировка искажения)`

The corresponding vulnerability pattern is:

- `shock (удар)` -> acute pressure but preserved bond architecture,
- `drift (дрейф)` -> selective bridge loss,
- `distortion (искажение)` -> harmful bond reorganization.

This is now one of the strongest current outputs of the Volume VI line.

## Minimal group-structured result

The next step moved the experiment from a flat small network into a minimal `group-structured network (сеть с групповой структурой)`.

At the current stage, this means:

- two groups,
- denser local within-group structure,
- and a small number of explicit inter-group `bridge bonds (мостовых связей)`.

### Why this matters

This step matters because Volume VI is not ultimately about one homogeneous mass of relations.

It is about fields in which:

- some relations are local,
- some are bridging,
- and crisis may damage those layers differently.

### Current result

The route-diagnostics line remains visible in the grouped version.

In particular:

- `shock route (маршрут удара)` remains strongly recoverable,
- `drift route (маршрут дрейфа)` remains weaker and bridge-sensitive,
- `distortion route (маршрут искажения)` remains pathological without intervention.

### Intervention result in grouped structure

The grouped version also strengthens the intervention story.

In the current prototype:

- `bridge preservation (сохранение мостовых связей)` improves `drift (дрейф)`,
- and `distortion blocking (блокировка искажения)` substantially improves `distortion (искажение)`.

This is important because the intervention line no longer works only in a flat toy architecture.

It now also works in a first structured multi-part field.

### Working interpretation

This grouped result suggests that the route-sensitive logic is not tied only to an overly simple topology.

It may survive the move toward:

- subgroup structure,
- local cohesion,
- and vulnerable inter-group bridges.

That makes the next step toward larger or mixed-route networks much more credible.

## Stepwise size result

The next scaling step tested whether the grouped route-diagnostics line survives a first increase in network size.

The current step compared:

- `n = 14`,
- `n = 28`,

while keeping the same basic grouped logic.

### What happened

The main route ordering survived the first size increase.

At `n = 28`:

- `shock route (маршрут удара)` remained the most recoverable,
- `drift route (маршрут дрейфа)` remained intermediate,
- `distortion route (маршрут искажения)` remained the most pathological.

So the route distinction did not disappear when the grouped system was enlarged.

### Intervention result at larger size

The intervention logic also remained meaningful at `n = 28`.

In particular:

- `bridge preservation (сохранение мостовых связей)` still improved `drift (дрейф)`,
- `distortion blocking (блокировка искажения)` still improved `distortion (искажение)`,
- and the grouped intervention effect did not vanish under this larger size.

### Working interpretation

This step suggests that the Volume VI route-diagnostics line is not limited to the smallest toy grouped case.

At least across the first scaling step:

- the qualitative ordering survives,
- the intervention logic survives,
- and the results are distorted quantitatively rather than destroyed structurally.

This is a good sign for further stepwise scaling.

## n = 100 plateau

The current scaling line has now reached `n = 100` in the grouped version.

This matters because the route-diagnostics architecture is no longer being tested only on small illustrative networks.

### What survived

At `n = 100`, the main route ordering still survives:

- `shock route (маршрут удара)` remains the cleanest recoverable route,
- `drift route (маршрут дрейфа)` remains intermediate,
- `distortion route (маршрут искажения)` remains the pathological route without intervention.

### What became stronger

One especially important result appeared at this scale:

- under intervention, `distortion route (маршрут искажения)` no longer leaves the earlier residual distorted remainder,
- and becomes much more cleanly recoverable than in the smaller grouped cases.

This suggests that the earlier residual floor may have been partly a small-scale structural effect rather than a permanent limit of the intervention logic.

### Current reading of the plateau

The line now has a meaningful working plateau:

- route distinction survives,
- grouped structure survives,
- scaling survives up to `n = 100`,
- and route-sensitive intervention remains effective at the larger scale.

This is a sufficiently strong checkpoint to justify the next step:

- making the model more metastable and more internally uneven rather than only larger.

## What this file should replace

When working quickly, this file should be the first place to look instead of reopening multiple small notes.

The smaller notes remain useful as process history, but this file should function as the main working checkpoint for the early route-diagnostics line of Volume VI.

## Current next step

The next likely step is not more note fragmentation.

It is either:

- a larger robustness sweep,
- or translation of this working result into a more manuscript-like argument once the route signatures stay stable under broader variation.
