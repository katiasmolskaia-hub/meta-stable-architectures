"""Baseline agent for narrow pilot experiments."""

from __future__ import annotations


class BaselineAgent:
    def __init__(self, capacity: float = 1.0, base_speed: float = 1.0, fail_margin: float = 0.25) -> None:
        self.capacity = capacity
        self.base_speed = base_speed
        self.fail_margin = fail_margin

    def solve(self, task_complexity: float, noise: float) -> dict[str, float | bool]:
        """
        Returns task outcome for one step.

        load = complexity + noise
        fail if load is far above capacity.
        time increases with load.
        """
        load = max(0.0, task_complexity + noise)
        overload = max(0.0, load - self.capacity)

        failed = overload > self.fail_margin
        solved = not failed

        # Time gets worse when overloaded.
        task_time = (1.0 / self.base_speed) * (1.0 + 0.9 * overload)

        return {
            "solved": solved,
            "failed": failed,
            "time": task_time,
            "overload": overload,
        }
