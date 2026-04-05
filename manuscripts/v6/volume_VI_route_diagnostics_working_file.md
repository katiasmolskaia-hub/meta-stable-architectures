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
- `outputs/volume_vi_bond_routes/shock_drift_state_sequence.csv`

The experiment currently compares three route types:

- `drift route (маршрут дрейфа)`,
- `shock route (маршрут удара)`,
- `distortion route (маршрут искажения)`.

It now also includes a first mixed case:

- `shock-drift mixed route (смешанный shock-drift маршрут)`.

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
- `shock-drift` has a visible but still recoverable tail,
- `drift (дрейф)` has a longer weakened tail,
- `distortion (искажение)` has a very large non-recoverable tail.

### 2. State-transition sequence

The routes are also distinguishable by the order of dominant bond-state changes.

Current dominant reading:

- `shock (удар)` passes through `crisis-support bond (кризисно-поддерживающую связь)` and returns to `stabilizing bond (стабилизирующей связи)`,
- `shock-drift` passes through crisis support but then falls into a longer `restored weak (восстановленно-слабый)` interval before stabilizing,
- `drift (дрейф)` passes through crisis support but falls back into weaker post-crisis states before returning only to `restored weak bond (восстановленной слабой связи)`,
- `distortion (искажение)` continues onward into `distorted bond (искаженную связь)`.

This is important because route diagnosis is now visible not only in how the system ends, but in how it passes through bond states.

## Process reading correction

The simulation line now suggests an important correction in how the process should be read.

The `field (поле)` should not be treated as if it performs the repair by itself.

It is better read as:

- a sensitive medium,
- a recorder of residual strain,
- and a condition for whether re-entry remains possible.

The more active process should instead be visible in:

- local bond break-and-repair,
- bridge strain and bridge reconnection,
- route memory,
- and only later in `RC / Instructor (RC / Инструкторе)` as bounded central gathering.

## Working stratification of the process

The newer reading also suggests that the Volume VI process should not be treated as if every variable lives on the same level of existence.

The process now looks better if read as stratified across:

- different temporal speeds,
- different process media,
- and different rights of entry into the cycle.

### Surface layer

This is the fast visible layer of the process.

It includes the variables that react first and fluctuate most directly:

- `effective_route`,
- `route_shock`,
- `route_drift`,
- `distortion`,
- `gatherability`,
- and much of early `local_rebond`.

This layer should be read as:

- what is happening now,
- what is directly felt,
- and what first becomes externally visible.

### Interface layer

This is the transition layer between local continuity and cross-group transfer.

It currently includes:

- `group_gap`,
- `interface_field`,
- `compatibility_window`,
- `bridge_drive`,
- `bridge_attempt_memory`,
- and early `bridge_rebond`.

This layer should be read as:

- not the bridge itself,
- but the condition under which transfer across difference may or may not become possible.

### Deep layer

This is the slower hidden reservoir of the process.

It currently includes:

- `field_memory`,
- `route_memory`,
- `compatibility_memory`,
- `bridge_failure_trace`,
- `bond_fatigue`,
- `bridge_fatigue`,
- and parts of `bridge_collapse`.

This layer should be read as:

- what the process carries for longer,
- what does not disappear when the visible phase changes,
- and what later alters the next cycle from below.

### Return flow

This is not yet fully built as an explicit architectural layer.

But conceptually it is already needed.

It would mean:

- that deep memory does not remain inert,
- but later returns into the surface and interface layers,
- changing the chance of compatibility,
- changing the shape of bridge recruitment,
- and changing whether late `RC` gathering becomes necessary.

So the process should now be read less as a flat network of variables and more as:

- `surface event (поверхностное событие)`,
- `interface transition (интерфейсный переход)`,
- `deep reservoir (глубинный резервуар)`,
- and `return flow (обратный возврат в новый цикл)`.

This matters because one important current limit may be that too many variables still live in one algorithmic tempo, even when they should belong to different process depths.

## First temporal-stratification result

The first explicit timing split between `surface` and `interface` layers now gives a small but important positive result.

The main change is not yet a full bridge recovery.

But it is the first cleaner sign that the process becomes more realistic when the interface layer is allowed to live more slowly than the immediate surface response.

Current reading:

- `interface_field` no longer behaves as if it were already fully formed at the earliest phase,
- `compatibility_window` becomes more visible and less purely instantaneous,
- `bridge_drive` remains weak but becomes more temporally legible,
- and `peak_bridge_rebonding_index` rises from the earlier near-vanishing band into a more visible `~2e-05` scale.

This should be read carefully.

It is not yet a success claim about bridge bonds.

It is a process claim:

- the system appears to react better when `surface event (поверхностное событие)` and `interface transition (интерфейсный переход)` do not live in exactly the same tempo.

So the first stratification check supports the broader hypothesis that the Volume VI process is not only networked, but also temporally layered.

### 3. State occupancy

The routes differ not only by transition order, but by how long they remain in different bond regimes.

The most useful measures here are:

- `dwell time in stabilizing state (время в стабилизирующем состоянии)`,
- `dwell time in restored-weak state (время в восстановленно-слабом состоянии)`,
- `dwell time in distorted state (время в искаженном состоянии)`,
- and `number of transitions (число переходов)`.

Current aggregate pattern:

- `shock (удар)` spends most of its post-crisis time in `stabilizing (стабилизирующем)` mode,
- `shock-drift` splits time between `crisis support (кризисной поддержкой)`, `restored weak (восстановленно-слабым)` mode, and later stabilization,
- `drift (дрейф)` spends much more time in `restored weak (восстановленно-слабом)` mode,
- `distortion (искажение)` spends a large final block in `distorted (искаженном)` mode.

## Current aggregate reading

The present repeated-run pattern suggests the following:

- `shock route (маршрут удара)` is an acute but recoverable route,
- `shock-drift mixed route (смешанный shock-drift маршрут)` is a mixed route with sharp entry and slower tail,
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
- intermediate tail and mixed state dwell for `shock-drift`,
- long restored-weak tail for `drift (дрейфа)`,
- large distorted dwell and failed re-entry for `distortion (искажения)`.

So the current route differences no longer look like one-off numerical accidents.

## First mixed-crisis result

The first mixed-crisis extension now tests `shock-drift (shock-drift)` under uneven group exposure.

This means that:

- one group can receive slightly earlier and stronger shock pressure,
- another group can receive a more delayed and stronger drift component,
- and the total crisis is no longer perfectly synchronous across the grouped field.

At `n=100`, the current aggregate reading is:

- `shock` remains the cleanest case with `mean tail_area = 0.000`,
- `shock-drift` remains recoverable with `mean final_gatherability = 0.941` and `mean tail_area = 0.710`,
- `drift` remains heavier with `mean final_gatherability = 0.912` and `mean tail_area = 1.244`,
- `distortion` remains pathological with `mean final_gatherability = 0.000` and `mean tail_area = 12.926`.

So the mixed route currently sits where it should:

- worse than `shock`,
- better than `drift`,
- and far healthier than `distortion`.

### State-sequence reading of the mixed route

The current canonical `shock-drift` state sequence is:

- `restored weak -> stabilizing -> crisis_support -> restored weak -> stabilizing`.

That is useful because it shows:

- acute mobilization does occur,
- but clean recovery is delayed,
- and the field spends a meaningful late interval in a weaker re-entry regime before stabilizing again.

### Uneven-group signal

The grouped mixed route also now produces a visible group-asymmetry signal.

The current `max_group_route_std` is about `0.263`.

This is modest rather than extreme.

But it is enough to show that:

- grouped structure can now hold a mixed route with uneven timing,
- without collapsing the route picture into unreadable noise.

## Hard mixed-crisis check

The next strengthening step tested a more severe `shock-drift mixed route (смешанный shock-drift маршрут)` under a harder uneven-group regime.

This harder regime increased:

- group timing asymmetry,
- delayed drift pressure,
- route-memory accumulation,
- and field-memory load on bridge structure.

The strongest current signal is that the internal memory variables now clearly rise:

- `final_route_memory_index (итоговый индекс памяти маршрута)` increases from about `0.906` to about `1.153`,
- `peak_field_memory_index (пиковый индекс памяти поля)` increases from about `0.253` to about `0.487`,
- `max_group_route_std (максимальная межгрупповая асимметрия маршрута)` increases from about `0.263` to about `0.426`.

So the field is no longer only passing through a mixed route.

It is beginning to retain the route as a real internal load.

### Honest limitation of the harder mixed regime

The harder mixed regime still does not yet produce as much prolonged damage as we would want.

At `n=100`, with intervention active:

- `mean final_gatherability` remains relatively high at about `0.937`,
- `mean tail_area` rises only to about `0.052`,
- `final_distorted` stays at `0`,
- while `collective_rebonding_index (индекс коллективного пересвязывания)` remains very strong at about `1.48`.

This suggests an important current limit:

- memory now forms,
- asymmetry now forms,
- but collective re-bonding is still too effective and too smooth.

In other words:

- the model now remembers the crisis route,
- but it still heals too cleanly once re-bonding begins.

## New process-mirror reading

The newer process-mirror line now sharpens this limit.

It shows that the main current problem is not only outcome quality.

It is process hierarchy.

At the present stage:

- local self-regulation can still become too successful,
- bridge activity is still too weak,
- and `RC` still enters too little and too late to count as a truly tested late coordinator.

This gives a clearer next target:

- make local bonds visible but non-omnipotent,
- make bridge bonds active under manageable difference,
- and let `RC` become necessary only when both of those layers no longer suffice.

## Current local conclusion on bond layers

The present cycle should be read as a local architectural check rather than a final theory claim.

At the current stage:

- `local bonds (локальные связи)` already behave as a real living layer,
- `route memory (память маршрута)` already leaves a strong readable trace,
- `field memory (память поля)` already works as a residual medium condition,
- but `bridge bonds (мостовые связи)` still express failure more clearly than successful reconnection.

In practical terms this means:

- bridge fatigue and bridge-failure trace are becoming visible,
- but bridge-driven reconnection is still too weak,
- and `RC / Instructor` still appears more as an emerging late reserve than as a fully tested late rescue layer.

So the current result is useful and honest:

- the local architecture is now readable,
- but the bridge layer still remains the main unfinished process layer.

## Seed-and-size stability check

The next local check tested whether this reading survives changes in random seed and network size.

The current `shock-drift` hard regime was checked at:

- `n = 56`,
- `n = 100`,
- `n = 140`,

with multiple seeds.

The main result is that the qualitative process picture remains stable:

- local re-bonding remains clearly active,
- route memory remains the dominant process lead,
- bridge recruitment remains extremely weak,
- and `RC / Instructor` remains a small late reserve rather than the main driver.

The current aggregate ranges are:

- `final_gatherability` stays roughly in the `0.930 - 0.951` band,
- `tail_area` stays roughly in the `0.385 - 0.580` band,
- `peak_local_rebonding_index` stays roughly in the `0.835 - 0.896` band,
- `peak_bridge_drive_index` stays near zero in the `0.0003 - 0.0009` band,
- `peak_central_fallback_index` stays small in the `0.0032 - 0.0084` band.

So the present local conclusion is no longer a one-seed accident.

It appears to be a stable property of the current architecture.

## Base-vs-current diagnostic cycle

The next diagnostic step compared:

- a simpler `base` mixed-crisis version,
- against the current more process-rich version,

under the same `shock-drift` crisis family.

### Single mixed crisis

Under a single mixed crisis, the current version does not look like a trivial overfit or a fake victory.

Instead, it shows a more disciplined tradeoff:

- final gatherability is slightly lower,
- local self-repair is less unrealistically dominant,
- the recovery tail becomes lighter,
- and late `RC` engagement becomes more visible.

So the present reading is:

- the newer process-rich version does not simply make everything better,
- it makes the process less falsely smooth.

### Repeated mixed crisis

Under repeated mixed crisis, both versions still remain heavily burdened.

The current version does not yet solve that burden.

But it again shows a more honest process profile:

- local self-repair weakens,
- bridge activity becomes slightly more visible,
- and `RC` engages more strongly than in the simpler base.

So the current conclusion is disciplined:

- the richer model is not yet a stronger rescue architecture,
- but it already looks like a more realistic crisis-process architecture.

## Expanded diagnostic package

The next diagnostic package compared:

- `base` vs `current`,
- `single` vs `repeated` mixed crisis,
- and `grouped` vs `ring` topology,

under the same `shock-drift` family.

### Main stable picture

Across all of these checks, one process pattern remains stable:

- `route memory` remains the dominant process lead,
- `local bonds` remain the strongest active self-regulation layer,
- `bridge recruitment` becomes visible but still remains weak,
- and `RC / Instructor` becomes more visible in the richer model, especially under repeated crisis and ring topology.

### Single-crisis comparison

Under single mixed crisis, the current richer model:

- slightly lowers final gatherability,
- clearly reduces the false smoothness of local self-repair,
- makes `RC` more visible,
- and does not produce a trivial across-the-board victory.

So the new model should not be read as a simple optimization.

It should be read as a more realistic process reading.

### Repeated-crisis comparison

Under repeated mixed crisis, both versions remain heavily burdened.

The current richer model does not solve that burden.

But it does make the internal process more legible:

- local re-bonding weakens,
- bridge-drive becomes more visible,
- and `RC` becomes noticeably more engaged than in the base.

### Topology comparison

The ring topology makes the same architecture harder.

Relative to the grouped topology, ring runs show:

- lower final gatherability,
- heavier tails,
- and more visible `RC` engagement.

That is useful because it suggests that the current process reading is not only a grouped-network artifact.

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
