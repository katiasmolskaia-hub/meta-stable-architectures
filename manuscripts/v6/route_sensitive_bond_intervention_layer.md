# Route-Sensitive Bond Intervention Layer

## Why this file matters

The current Volume VI diagnostics line no longer only distinguishes crisis routes.

It now begins to indicate which part of the bond architecture is most at risk under each route.

This creates the need for a new layer:

- not only diagnosis,
- but route-sensitive bond intervention.

This file gives the first working skeleton for that layer.

## Core idea

The intervention logic should not be:

- strengthen all bonds,
- preserve all bonds equally,
- or apply one generic recovery mechanism.

Instead, it should be:

- identify the likely crisis route,
- identify the bond-class vulnerability pattern,
- and apply the smallest useful intervention that preserves re-entry.

So the goal is not maximal control.

The goal is selective support of the relational architecture.

## Minimal intervention principle

The intervention layer should answer one question:

- what kind of bond support is needed under this route so that the field remains re-gatherable?

This means that intervention should depend on:

- crisis route,
- bond class,
- current bond state,
- and current re-entry potential.

## Route 1: Shock intervention

### Route pattern

`Shock route (маршрут удара)` is currently read as:

- acute,
- high-pressure,
- but still recoverable.

The field is hit quickly, but the network can still reorganize if temporary support is recruited in time.

### Main intervention idea

The main intervention here should be:

- `temporary support scaffold (временный поддерживающий каркас)`.

### What it should do

It should:

- temporarily raise usable stabilizing capacity,
- accelerate transition into `crisis-support bonds (кризисно-поддерживающие связи)`,
- preserve enough `stability (устойчивости)` to avoid collapse,
- and then relax back toward `stabilizing bonds (стабилизирующим связям)` rather than keeping the field under pressure.

### What it should not do

It should not:

- permanently harden all bonds,
- or over-fix the field after the acute phase has passed.

## Route 2: Drift intervention

### Route pattern

`Drift route (маршрут дрейфа)` is currently read as:

- slow weakening,
- long tail,
- and selective loss of vulnerable `bridge bonds (мостовых связей)`.

### Main intervention idea

The main intervention here should be:

- `bridge preservation / bridge repair (сохранение и восстановление мостовых связей)`.

### What it should do

It should:

- protect `bridge bonds (мостовые связи)` from slow functional emptying,
- reduce chronic erosion of `bond viability (жизнеспособности связи)`,
- preserve enough `bond freedom (свободы связи)` that bridges remain reactivatable,
- and stop weakened bridges from sliding into latent or distorted states too early.

### What it should not do

It should not:

- simply maximize rigidity,
- because drift does not primarily destroy the field through acute impact.

It erodes pathways of return.

## Route 3: Distortion intervention

### Route pattern

`Distortion route (маршрут искажения)` is currently read as:

- harmful relational reorganization,
- high `distortion load (нагрузка искажения)`,
- and failed `re-entry (повторный вход)`.

### Main intervention idea

The main intervention here should be:

- `distortion blocking (блокировка искажения)`.

### What it should do

It should:

- suppress harmful bond activation,
- reduce spread of bad local structure,
- limit transitions into `distorted bonds (искаженные связи)`,
- and preserve any remaining viable support scaffold.

### What it should not do

It should not:

- treat all active bonds as good,
- or mistake persistence for health.

In this route, some active bonds are actively harmful.

## Minimal intervention variables

At the first stage, the intervention layer could be represented by three compact controls:

- `I_shock(t)` = temporary support intensity for acute crisis,
- `I_bridge(t)` = bridge-preservation intensity,
- `I_block(t)` = distortion-blocking intensity.

These should not all be active at once at equal strength.

The route-diagnostics layer should determine which control becomes dominant.

## Minimal intervention rule

At the working-concept level, the rule is:

- if route is `shock-like (удароподобный)`, increase `I_shock`,
- if route is `drift-like (дрейфоподобный)`, increase `I_bridge`,
- if route is `distortion-like (искаженно-патологический)`, increase `I_block`.

Later versions may allow mixed or uncertain diagnosis.

But the first intervention model should stay simple.

## Relation to bond balance

This layer matters because different routes require different bond balances.

For example:

- `shock (удар)` needs temporary support without long over-hardening,
- `drift (дрейф)` needs preservation of flexible pathways,
- `distortion (искажение)` needs suppression of harmful bond dynamics.

So intervention should not be read as a generic increase of connection.

It should be read as route-sensitive adjustment of:

- `bond freedom (свободы связи)`,
- `bond stability (устойчивости связи)`,
- and `distortion control (контроля искажения)`.

## Minimal implementation direction

The first simulation version of this layer should remain small.

It should:

- use the existing route diagnosis,
- add one intervention mode per route,
- and test whether the intervention improves the vulnerable part of the bond architecture without damaging the rest.

## Minimal conclusion

The route-diagnostics line now naturally leads to a route-sensitive intervention layer.

The key principle is simple:

- not all crises need the same relational help,
- and not all bonds should be helped in the same way.

Volume VI can therefore move from:

- route diagnosis alone

toward:

- route-sensitive preservation of re-entry through selective bond intervention.
