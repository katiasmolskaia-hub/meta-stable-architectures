"""Meta-layer v1 agent with continuous isolation I(t) dynamics."""

from __future__ import annotations


class MetaLayerAgentV1:
    def __init__(
        self,
        capacity: float = 1.0,
        base_speed: float = 1.0,
        fail_margin: float = 0.25,
        tau: int = 6,
        # Crisis threshold parameters
        c0: float = 0.08,
        mu_k: float = 0.25,
        zeta: float = 0.20,
        # Adaptation parameters
        nu: float = 0.2,
        delta_k: float = 0.3,
        xi: float = 2.0,
        # Isolation effect
        damp_strength: float = 0.60,
        speed_penalty: float = 1.20,
    ) -> None:
        self.capacity = capacity
        self.base_speed = base_speed
        self.fail_margin = fail_margin

        self.tau = max(1, tau)
        self.c0 = c0
        self.mu_k = mu_k
        self.zeta = zeta
        self.nu = nu
        self.delta_k = delta_k
        self.xi = xi
        self.damp_strength = damp_strength
        self.speed_penalty = speed_penalty

        # Meta-layer state
        self.crisis_history: list[float] = []
        self.k_struct = 0.0
        self.i_iso = 0.05

    def _update_threshold(self, crisis_energy: float) -> float:
        self.crisis_history.append(crisis_energy)
        if len(self.crisis_history) > self.tau:
            self.crisis_history.pop(0)

        c_avg = sum(self.crisis_history) / len(self.crisis_history)
        c_crit = self.c0 * (2.718281828 ** (-self.mu_k * self.k_struct)) + self.zeta * c_avg
        return c_crit

    def solve(self, task_complexity: float, noise: float) -> dict[str, float | bool]:
        # Raw load
        raw_load = max(0.0, task_complexity + noise)
        raw_overload = max(0.0, raw_load - self.capacity)

        # Crisis energy scaled by failure margin
        scale = max(self.fail_margin, 1e-6)
        c = (raw_overload / scale) ** 2
        c_crit = self._update_threshold(c)

        # Update structural knowledge
        self.k_struct = max(0.0, self.k_struct + (self.nu * c - self.delta_k * self.k_struct))

        # Update isolation I(t) (logistic bounded)
        di = self.xi * (c - c_crit) * self.i_iso * (1.0 - self.i_iso)
        self.i_iso = max(0.0, min(1.0, self.i_iso + di))

        # Isolation reduces effective load
        effective_load = raw_load * (1.0 - self.damp_strength * self.i_iso)
        overload = max(0.0, effective_load - self.capacity)

        failed = overload > self.fail_margin
        solved = not failed

        # Speed penalty increases with isolation
        task_time = (1.0 / self.base_speed) * (1.0 + 0.9 * overload) * (1.0 + (self.speed_penalty - 1.0) * self.i_iso)

        return {
            "solved": solved,
            "failed": failed,
            "time": task_time,
            "overload": overload,
            "I": self.i_iso,
            "C": c,
            "Ccrit": c_crit,
        }


class MetaLayerAgentV1Soft(MetaLayerAgentV1):
    """Softer variant: less aggressive isolation, more permissive threshold."""

    def __init__(self) -> None:
        super().__init__(
            c0=0.12,
            zeta=0.30,
            xi=1.0,
            damp_strength=0.40,
            speed_penalty=1.10,
        )
