# Large-Scale 1000 Plateau

This note records the current `N = 1000` validation point for the Volume V line.

## 1. Why this note matters

Earlier validation at:
- `N = 100`,
- `N = 300`

already suggested that the current architecture did not collapse under scaling.

The next serious question was whether that remained true at:
- `N = 1000`

This note records that answer.

## 2. What was compared

The comparison used the same heterogeneous lesson cycle:
- noise,
- lag,
- contagion,
- mixed heavy lesson.

The two model variants were:

### A. Baseline multimode

This includes:
- adaptive coordination,
- three-channel routing,
- multimode instructor signal,

but no demper.

### B. Full model with demper

This includes:
- adaptive coordination,
- three-channel routing,
- multimode instructor signal,
- buffered recovery / demper.

Heterogeneity was preserved at large scale through three cohorts with different:
- response parameters,
- noise levels,
- crisis thresholds,
- and initial attunement.

So this was not a flat-network scaling test.

## 3. Main `N = 1000` result

The effect held.

At `N = 1000`:

### Baseline multimode
- mean recovery time: `1.56`
- mean tail span: `2.735`
- mean peak error: `0.260`
- mean peak turbulence: `0.260`
- mean final attunement: `0.737`

### Full model with demper
- mean recovery time: `1.205`
- mean tail span: `2.355`
- mean peak error: `0.260`
- mean peak turbulence: `0.259`
- mean final attunement: `0.739`

## 4. Interpretation

This is a strong result because it shows that the current advantage:
- is not a small-network artifact,
- does not disappear when the system becomes much larger,
- and does not require sacrificing the field-quality gains.

In practical terms:
- the timing advantage survives,
- the tail advantage survives,
- and the coordination quality remains essentially intact.

## 5. Whole-line significance

At this stage, the current Volume V architecture has support on several levels:

### A. Internal structure

- mutual attunement,
- three-channel orientation,
- multimode instruction.

### B. Long-horizon improvement

- long curriculum accumulates field-quality gains.

### C. Large-scale validation

- buffered recovery still improves recovery time and tail at `N = 1000`.

This makes the current line much harder to dismiss as a fragile or overly local phenomenon.

## 6. Practical conclusion

At the `N = 1000` plateau, the project can now say:

- the full model remains coherent at large scale,
- the demper advantage remains visible,
- and the system keeps both:
  - coordination quality,
  - and improved coarse timing metrics.

This is one of the strongest validation points reached so far in Volume V.
