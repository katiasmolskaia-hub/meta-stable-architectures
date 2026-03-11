"""Meta-layer v0 agent with crisis-aware modulation."""

from __future__ import annotations


class MetaLayerAgentV0:
    def __init__(
        self,
        capacity: float = 1.0,
        base_speed: float = 1.0,
        fail_margin: float = 0.25,
        tau: int = 5,
        trigger: float = 0.18,
        damp_strength: float = 0.55,
    ) -> None:
        self.capacity = capacity
        self.base_speed = base_speed
        self.fail_margin = fail_margin

        # Meta-layer state
        self.tau = max(1, tau)
        self.trigger = trigger
        self.damp_strength = damp_strength
        self.crisis_history: list[float] = []
        self.crisis_mode = False

    def _update_crisis(self, overload: float) -> float:
        self.crisis_history.append(overload)
        if len(self.crisis_history) > self.tau:
            self.crisis_history.pop(0)

        avg_crisis = sum(self.crisis_history) / len(self.crisis_history)
        self.crisis_mode = avg_crisis > self.trigger
        return avg_crisis

    def solve(self, task_complexity: float, noise: float) -> dict[str, float | bool]:
        # Raw load
        raw_load = max(0.0, task_complexity + noise)
        raw_overload = max(0.0, raw_load - self.capacity)
        avg_crisis = self._update_crisis(raw_overload)

        # In crisis mode we trade speed for stability.
        if self.crisis_mode:
            effective_load = raw_load * (1.0 - self.damp_strength)
            speed_penalty = 1.25
        else:
            effective_load = raw_load
            speed_penalty = 1.0

        overload = max(0.0, effective_load - self.capacity)
        failed = overload > self.fail_margin
        solved = not failed

        task_time = (1.0 / self.base_speed) * (1.0 + 0.9 * overload) * speed_penalty

        return {
            "solved": solved,
            "failed": failed,
            "time": task_time,
            "overload": overload,
            "crisis_mode": self.crisis_mode,
            "avg_crisis": avg_crisis,
        }
