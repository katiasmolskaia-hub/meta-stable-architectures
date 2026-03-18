# Network Extension: Coupled Phases with Cascading Isolation

## Motivation
Large networks of adaptive agents can fail due to loss of phase coherence under stress. The novel contribution here is **dynamic decoupling**: when a node enters crisis, its isolation variable increases and **reduces coupling to neighbors**, preventing network-wide contagion.

This extension is intended as a minimal, testable bridge from the single-agent model to networked systems.

## Core Idea
Each node carries a phase variable and a local crisis/isolation state. Weak neighbor coupling promotes coherence, while isolation reduces coupling during crisis.

When a node is in crisis (high local stress), its isolation variable increases, attenuating both its influence on neighbors and its susceptibility to their phases. After recovery, coupling is restored and the node re-locks to the network rhythm.

## Model (Kuramoto-style with isolation-gated coupling)
Let node phases be `theta_k(t)`.

**Phase dynamics**
```
d theta_k/dt = omega_k
              + (1 - I_k) * (K/N) * sum_j sin(theta_j - theta_k)
```

**Crisis / isolation coupling**
```
C_k = |dV_k/dt| or a stress proxy from local dynamics
Ccrit_k = C0 * exp(-mu * K_k) + zeta * avg(C_k)
d I_k/dt = xi * (C_k - Ccrit_k) * I_k * (1 - I_k)
```

**Key property**
Isolation gates coupling: when `I_k -> 1`, node k effectively decouples from neighbors.

## Minimal Simulation (prototype)
The following pseudocode is sufficient to test the mechanism:

```
for t in timesteps:
  for k in nodes:
    # phase update
    theta_k += dt * (
      omega_k
      + (1 - I_k) * (K/N) * sum_j sin(theta_j - theta_k)
    )

    # crisis proxy from local energy or stress
    C_k = stress(theta_k, ...)
    Ccrit_k = C0 * exp(-mu * K_k) + zeta * avg(C_k)
    I_k += dt * xi * (C_k - Ccrit_k) * I_k * (1 - I_k)
```

## What to Look For
1. **Phase coherence:** dispersion of phases decreases except for isolated nodes.
2. **Isolation effect:** nodes in crisis decouple, recover, and re-lock to the network rhythm.
3. **Optional ablation:** add a weak global forcing term to test whether exogenous rhythm improves robustness.

## Interpretation
Isolation-gated coupling provides a direct, testable network-level extension of the metastable architecture. A weak global forcing signal can be used as an ablation, not as a required component.
