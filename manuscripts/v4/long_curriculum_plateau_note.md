# Long Curriculum Plateau

This note records the current result of the long curriculum branch in Volume V.

## 1. What was tested

The main question was whether longer shared development would finally move the coarse timing metrics:
- recovery time,
- tail span.

To test this, a longer curriculum was introduced:
- 16 lessons,
- 4 blocks,
- repeated cycle of `noise -> lag -> contagion -> mix`,
- transfer of:
  - agent attunement,
  - group memory,
  - coordinator phase,
- using the current best coordination setting:
  - adaptive three-channel routing,
  - multimode instructor signaling.

## 2. Main outcome

The long curriculum produced a strong and stable improvement in field quality.

Across blocks:
- peak error fell substantially,
- peak turbulence fell substantially,
- attunement rose toward saturation.

At the same time:
- mean recovery time remained essentially unchanged,
- mean tail span remained essentially unchanged.

## 3. Numerical summary by block

### Block 1
- mean recovery time: `1.43`
- mean tail span: `2.745`
- mean peak error: `0.258`
- mean peak turbulence: `0.256`
- mean attunement: `0.751`

### Block 2
- mean recovery time: `1.43`
- mean tail span: `2.745`
- mean peak error: `0.208`
- mean peak turbulence: `0.207`
- mean attunement: `0.960`

### Block 3
- mean recovery time: `1.43`
- mean tail span: `2.745`
- mean peak error: `0.200`
- mean peak turbulence: `0.199`
- mean attunement: `0.979`

### Block 4
- mean recovery time: `1.43`
- mean tail span: `2.745`
- mean peak error: `0.199`
- mean peak turbulence: `0.199`
- mean attunement: `0.981`

## 4. Interpretation

This is an important result.

It shows that:
- repeated shared lessons are not empty,
- the group does become cleaner and less turbulent over time,
- the coordinator-group relation becomes more stable,
- field quality accumulates as a real long-horizon effect.

However, the result also shows that the current model does not yet convert that improved field quality into:
- faster post-stress recovery,
- or shorter recovery tails.

So the present system supports:
- better coordination quality,
- but not yet better speed metrics.

## 5. What this means conceptually

The long curriculum result strengthens the earlier claim of Volume V:

`mutual development between instructor and group improves field quality first`

It also sharpens the limit:

`coarse timing metrics appear to be harder and may require a deeper or richer model than the current one`

This is not a failure.
It is a separation of layers:
- quality improves first,
- speed may require another mechanism or a longer developmental horizon.

## 6. Role of memory

In this long curriculum branch, `K_g` remains weak.
It does not emerge as the main driver of improvement.

The main driver is still:
- attunement,
- better channel use,
- and better multimode coordination.

So the current evidence still supports:
- agent-side and interaction-side development first,
- memory as secondary.

## 7. Why this matters

There is an understandable temptation to treat time-to-recovery as the most impressive hook.
That is the metric many readers will look for first.

But the current results suggest that the more honest and technically grounded contribution is different:

- the system learns to become less noisy,
- less turbulent,
- and easier to coordinate over time,
- even before it becomes faster.

That is already a meaningful systems result.

## 8. Practical conclusion

At this plateau, the project can now state:

- multimode instruction improves field quality across heterogeneous lessons;
- long curriculum makes that quality improvement accumulate over time;
- coarse recovery-time and tail metrics remain resistant at the current model depth.

So the current emphasis should remain:
- quality,
- coherence,
- turbulence reduction,
- and long-horizon mutual development.

Not yet:
- strong speed claims,
- or strong tail-compression claims.
