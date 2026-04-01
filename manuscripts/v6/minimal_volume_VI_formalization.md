# Minimal Volume VI Formalization

## Why this file matters

Volume VI already contains a broad conceptual architecture.

What is needed now is not a full formalization of every idea at once, but a minimal formal block that can be tested without overloading the model.

The purpose of this file is therefore:

- to define the first testable slice of Volume VI,
- to separate the new bond-centered logic from the earlier instructor-centered logic,
- and to clarify what should and should not enter the first formal experiments.

## Core decision

The first formal step of Volume VI should not attempt to include the entire inherited architecture from Volume V.

Instead, it should isolate the genuinely new question:

- how crisis routes reshape bond states,
- and how those bond states preserve or destroy the possibility of re-entry.

This keeps the first model disciplined.

## What the first model is trying to explain

The first model should explain four things:

1. the same visible crisis may be reached through different routes,
2. different routes may push bonds into different functional states,
3. those bond states affect whether the field remains gatherable,
4. and re-entry depends on the bond architecture, not only on crisis intensity.

This is the minimal formal heart of Volume VI.

## What should be included

The first formal block should include:

- a minimal field-level variable for `gatherability (собираемость)`,
- bond-level variables,
- a route signal,
- and a derived measure of `re-entry potential (потенциал повторного входа)`.

The recommended minimal variables are:

- `B_{ij}` = `bond viability (жизнеспособность связи)`,
- `F_{ij}` = `bond freedom (свобода связи)`,
- `S_{ij}` = `bond stability (устойчивость связи)`,
- `D_{ij}` = `distortion load (нагрузка искажения)`,
- `R(t)` = `route pressure (давление маршрута)`,
- `G(t)` = `gatherability (собираемость поля)`,
- `E_{ij}` = `re-entry support (поддержка повторного входа)`.

## What should not be included yet

The first formal block should not yet include the full Volume V mechanism stack.

In particular, it should temporarily leave out:

- full `multimode instructor (мультимодальный инструктор)`,
- advanced `recognition (распознавание)`,
- `demper (демпфер)`,
- long-horizon memory accumulation,
- and richer field-reading logic beyond what is minimally needed.

This is not because those elements are unimportant.

It is because the first Volume VI formal test should isolate the new mechanism rather than hide it inside an already complex architecture.

## Relation to Volume V

Volume V centered on instructor-side adaptation:

- the instructor reads the field,
- changes mode,
- improves attunement,
- and supports recovery through bounded soft intervention.

Volume VI should begin from a different center:

- the field itself reorganizes through bond-state dynamics,
- crisis routes alter the relational structure,
- and re-entry depends on which bonds remain living, empty, stabilizing, distorted, or recoverable.

So the first formal model of Volume VI should inherit the soft field-sensitive philosophy of Volume V, but not the whole instructor machinery.

## Minimal dynamical proposal

At the first stage, the bond dynamics may be represented in a compact way:

```math
\dot B_{ij}
=
\alpha_1 S_{ij}
+
\alpha_2 F_{ij}
+
\alpha_3 G
-
\alpha_4 D_{ij}
-
\alpha_5 R
```

```math
\dot F_{ij}
=
\beta_1(1-F_{ij})A_{ij}
-
\beta_2 S_{ij}F_{ij}
-
\beta_3 D_{ij}F_{ij}
```

```math
\dot S_{ij}
=
\gamma_1 B_{ij}
+
\gamma_2 G
-
\gamma_3 F_{ij}^2
-
\gamma_4 D_{ij}
```

```math
\dot D_{ij}
=
\delta_1 R
+
\delta_2 C_{ij}
-
\delta_3 S_{ij}
-
\delta_4 B_{ij}
```

and a compact measure of re-entry support:

```math
E_{ij}
=
B_{ij}(\lambda_1 S_{ij} + \lambda_2 F_{ij} - \lambda_3 D_{ij})
```

These equations should be treated only as a starting operational scaffold.

## Minimal route structure

At the first stage, `route pressure (давление маршрута)` should be implemented in the simplest possible way through three broad regimes:

- `drift route (маршрут дрейфа)`,
- `shock route (маршрут удара)`,
- `distortion route (маршрут искажения)`.

The first test should ask whether these routes push bond variables differently even when the visible crisis severity looks similar.

## What counts as success

The first formal stage should be judged successful if it shows the following:

- different crisis routes lead to meaningfully different bond-state trajectories,
- the same visible field crisis can conceal different relational histories,
- re-entry potential depends on bond-state configuration rather than on crisis size alone,
- and the model can distinguish between at least:
  - `functionally empty bond (функционально пустая связь)`,
  - `stabilizing bond (стабилизирующая связь)`,
  - `distorted bond (искаженная связь)`,
  - and `restored weak bond (восстановленная слабая связь)`.

## What counts as failure

The first formal stage should be reconsidered if:

- all crisis routes produce nearly identical bond trajectories,
- re-entry depends only on crisis intensity and not on relational state,
- freedom and stability collapse into the same variable in practice,
- or the equations become too complex to interpret before they produce any clear new result.

## Practical strategy

The right strategy is:

1. keep the model small,
2. test the route-to-bond logic first,
3. test whether bond-state interpretation is readable,
4. and only then reintroduce richer instructor-side architecture if needed.

This preserves the new identity of Volume VI while keeping the formal work experimentally manageable.

## Minimal conclusion

The first formalization of Volume VI should not attempt to formalize the whole volume.

It should formalize one central claim:

- different crisis routes reshape bond states differently,
- and this difference helps determine whether the field can re-enter coherence.

That is the cleanest and most defensible formal starting point.
