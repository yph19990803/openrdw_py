from __future__ import annotations

import argparse
from pathlib import Path

from .exporters import export_trace_csv
from .factory import build_scheduler, SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Python OpenRDW simulation")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--redirector", default="s2c")
    parser.add_argument("--resetter", default="two_one_turn")
    parser.add_argument("--output", default="python_openrdw/out/demo_trace.csv")
    args = parser.parse_args()

    scheduler = build_scheduler(
        SimulationConfig(
            redirector=args.redirector,
            resetter=args.resetter,
            agent_count=1,
        )
    )
    agents = scheduler.run(args.steps)
    output = Path(args.output)
    export_trace_csv(output, agents[0].trace)
    print(f"Wrote {len(agents[0].trace)} steps to {output}")


if __name__ == "__main__":
    main()
