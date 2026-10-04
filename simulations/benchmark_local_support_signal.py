"""Sequential runtime and separately traced peak memory, local controller only."""
import gc
import json
import time
import tracemalloc

import numpy as np

from experiment_local_support_signal import OUT, SIZES, Params, simulate


def main():
    # Keep diagnostic request history disabled: it belongs to the shuffled
    # offline control, not to an operating node's bounded memory.
    simulate("local_signal", "repeated_damage", 10200, Params(n=96), keep_log=False)
    rows = []
    for n in SIZES:
        durations = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            r, log = simulate("local_signal", "repeated_damage", 10200, Params(n=n), keep_log=False)
            durations.append(time.perf_counter() - start)
            assert not log
        gc.collect()
        tracemalloc.start()
        simulate("local_signal", "repeated_damage", 10200, Params(n=n), keep_log=False)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(dict(n=n, policy="local_signal", scenario="repeated_damage", seed=10200,
                         wall_seconds=durations, wall_median_seconds=float(np.median(durations)),
                         traced_peak_bytes=peak, peak_bytes_per_node=peak/n,
                         allowed_edges=r["allowed_edges"], requests=r["requests"], messages=r["messages"]))
        print(json.dumps(rows[-1]), flush=True)
    (OUT / "benchmark.json").write_text(json.dumps(dict(
        description="One process; three timings without tracer; separate tracemalloc run; no request log; not process RSS or robot energy",
        results=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
