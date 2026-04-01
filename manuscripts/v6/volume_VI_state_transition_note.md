# Volume VI State Transition Note

## Why this file matters

The second round of the first Volume VI experiment added explicit tracking of dominant bond-state transitions through time.

This matters because the route distinction is no longer visible only in final outcomes or recovery tails.

It is now also visible in the sequence of bond-state changes.

## Current transition sequences

The current prototype produces the following dominant state sequences.

### Drift route

`sustained weak bond (поддерживаемая слабая связь)`
-> `restored weak bond (восстановленная слабая связь)`
-> `stabilizing bond (стабилизирующая связь)`
-> `crisis-support bond (кризисно-поддерживающая связь)`
-> `restored weak bond (восстановленная слабая связь)`
-> `latent weak bond (латентная слабая связь)`
-> `restored weak bond (восстановленная слабая связь)`

### Shock route

`sustained weak bond (поддерживаемая слабая связь)`
-> `restored weak bond (восстановленная слабая связь)`
-> `stabilizing bond (стабилизирующая связь)`
-> `crisis-support bond (кризисно-поддерживающая связь)`
-> `restored weak bond (восстановленная слабая связь)`
-> `stabilizing bond (стабилизирующая связь)`

### Distortion route

`sustained weak bond (поддерживаемая слабая связь)`
-> `restored weak bond (восстановленная слабая связь)`
-> `stabilizing bond (стабилизирующая связь)`
-> `crisis-support bond (кризисно-поддерживающая связь)`
-> `restored weak bond (восстановленная слабая связь)`
-> `latent weak bond (латентная слабая связь)`
-> `distorted bond (искаженная связь)`

## What this already suggests

Several early interpretations are now possible.

### 1. Shock tends toward re-stabilization

The `shock route (маршрут удара)` does not remain trapped in a weakened state.

It passes through a `crisis-support (кризисно-поддерживающий)` phase and then returns to a `stabilizing (стабилизирующему)` regime.

This is consistent with an acute but recoverable route.

### 2. Drift tends toward prolonged weakening

The `drift route (маршрут дрейфа)` does not collapse into `distorted bonds (искаженные связи)` in the current prototype.

However, it does not cleanly restabilize either.

Instead, it falls back through `latent weak bonds (латентные слабые связи)` and returns only to a `restored weak (восстановленно-слабому)` regime.

This is consistent with a longer weakened tail.

### 3. Distortion becomes qualitatively pathological

The `distortion route (маршрут искажения)` is the only one that continues from weakening into a final `distorted (искаженный)` regime.

This supports the idea that not all crisis routes merely weaken the field.

Some may actively reorganize it in a harmful direction.

## Relation to tail diagnostics

These transition sequences fit the new `tail-sensitive diagnostics (диагностике хвостов)`.

The current interpretation is:

- `shock (удар)` shows a short recoverable tail,
- `drift (дрейф)` shows a long weakened tail,
- `distortion (искажение)` shows a non-recoverable tail.

So the project now has two aligned forms of distinction:

- state-transition distinction,
- and tail-shape distinction.

## What remains open

These results are still first-round evidence rather than a finished claim.

The next important step is to measure:

- how often each transition occurs,
- how long the system remains in each state,
- and how stable these signatures remain across repeated runs and parameter variation.

## Minimal conclusion

The current prototype already suggests that different crisis routes are not distinguished only by endpoint quality.

They are also distinguished by:

- the order of bond-state transitions,
- the ability or inability to return to stabilization,
- and the shape of the recovery tail that follows crisis.
