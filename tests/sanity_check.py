"""
Optional sanity test.
Run:
    python tests/sanity_check.py
from repo root (after installing requirements).
"""

import os
import subprocess
import sys
#####################################################################
def main() -> None:
    cmd = [
        sys.executable,
        "run_agent.py",
        "--data", "data/example_dataset.csv",
        "--target", "auto",
        "--quiet",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)                                  # Capture output so we can inspect the generated folder

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise SystemExit("Sanity check failed.")

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]             # Ignore empty output lines
    if not stdout_lines:
        raise SystemExit("Sanity check failed: run_agent.py did not print an output directory.")

    out_dir = stdout_lines[-1]                                                                        # run_agent.py prints the final output directory on the last line
    print("Output dir:", out_dir)

    if not os.path.isdir(out_dir):
        raise SystemExit(f"Sanity check failed: output directory does not exist -> {out_dir}")

    expected = [
        "report.md",
        "metrics.json",
        "reflection.json",
        "eda_summary.json",
        "plan.json",
        "confusion_matrix.png",
    ]  # Core artefacts expected from a successful classification run

    missing = [fname for fname in expected if not os.path.exists(os.path.join(out_dir, fname))]
    if missing:
        raise SystemExit(f"Missing expected outputs: {missing}")

    print("Sanity check passed.")
#####################################################################
if __name__ == "__main__":
    main()
