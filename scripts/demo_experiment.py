"""Task 8 demo: run the multi-start experiment and print the summary table."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsn.config import DEFAULT_CONFIG
from wsn.experiment import format_summary_table, run_experiment


def main() -> None:
    # A smaller, quicker config for a readable demo.
    config = DEFAULT_CONFIG.with_overrides(n_starts=5)
    summaries = run_experiment(config)

    print("=" * 72)
    print(f"Multi-start comparison  (N={config.n_sensors}, starts={config.n_starts})")
    print("=" * 72)
    print(format_summary_table(summaries))
    print("-" * 72)
    for method, s in summaries.items():
        br = s.best_record
        print(
            f"{method}: best coverage {s.best_coverage:.3f} "
            f"(seed {br.seed}, connected={br.connected})"
        )


if __name__ == "__main__":
    main()
