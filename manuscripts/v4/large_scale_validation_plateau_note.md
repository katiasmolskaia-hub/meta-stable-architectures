# Large-Scale Validation Plateau

This note records the current result of the first large-scale validation step in Volume V.

## 1. Why this step mattered

After the demper branch, the model finally showed clear gains in:
- recovery time,
- recovery tail.

At that point, the next important question was no longer whether the mechanism worked only on a small network.
The real question became:

`Does the effect survive when the network becomes substantially larger?`

## 2. What was tested

The validation compared two model variants:

- baseline multimode coordination without demper,
- full model with demper.

The test used the same lesson cycle:
- noise,
- lag,
- contagion,
- mixed heavy lesson.

The comparison was run at:
- `N = 100`,
- `N = 300`.

Importantly, heterogeneity was preserved.
The network did not become uniform at larger size.
Instead, it still contained three cohorts with different:
- response parameters,
- noise levels,
- crisis thresholds,
- and initial attunement levels.

So this was a genuine scaling test of a heterogeneous system, not a scaling test of a flattened one.

## 3. Main result

The effect held.

At both `N = 100` and `N = 300`, the full model with demper preserved the same qualitative advantage:
- lower recovery time,
- shorter recovery tail,
- with only negligible change in field error and turbulence.

## 4. Numerical summary

### N = 100

Baseline multimode:
- mean recovery time: `1.56`
- mean tail span: `2.74`
- mean peak error: `0.259`
- mean peak turbulence: `0.258`

Full model with demper:
- mean recovery time: `1.205`
- mean tail span: `2.35`
- mean peak error: `0.259`
- mean peak turbulence: `0.257`

### N = 300

Baseline multimode:
- mean recovery time: `1.56`
- mean tail span: `2.74`
- mean peak error: `0.259`
- mean peak turbulence: `0.259`

Full model with demper:
- mean recovery time: `1.205`
- mean tail span: `2.36`
- mean peak error: `0.259`
- mean peak turbulence: `0.258`

## 5. Interpretation

This is a strong validation result.

It shows that the current mechanism stack does not rely on a tiny-network artifact.

The effect of the full model:
- does not collapse when the network grows,
- does not depend on flattening heterogeneity,
- and does not require paying for timing gains with obvious degradation of field quality.

In simple terms:
- the model still works when there are many more agents,
- and the agents are still meaningfully different.

## 6. What this means structurally

At this point, the current Volume V line has three levels of support:

### A. Local / conceptual support

- attunement improves field quality,
- three-channel reference structure is meaningful,
- multimode instruction improves heavy lessons.

### B. Long-horizon support

- long curriculum accumulates better field quality over time.

### C. Coarse-timing support

- demper improves recovery time and tail,
- and that timing advantage survives scaling to larger heterogeneous networks.

This makes the overall model much more credible as a whole system.

## 7. Practical conclusion

At this plateau, the project can now say:

- the current full model is not only conceptually coherent;
- it is also robust enough to preserve its main timing advantage at larger `N`.

This is the first point at which the model begins to look less like a delicate small-network demonstration and more like a scalable system-level architecture.

## 8. Immediate next step

The natural next test after this plateau is:
- `N = 1000`

That step is no longer exploratory in the same way.
It is now a serious stress test of whether the current architecture keeps its gains at a much larger scale.
