# orchestrate_kpop_pipeline.py
# Runs the 5-step kpop pipeline in strict sequence with status + error handling.

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# -------- Config --------
# Put this orchestrator in the same folder as the five scripts, or adjust BASE_DIR.
try:
    BASE_DIR = Path(__file__).parent.resolve()
except NameError:
    # Fallback when running in notebooks or REPL
    BASE_DIR = Path.cwd().resolve()

SCRIPTS = [
    ("1) Harvest aliases (hot-only)", "1_kpop_all_criteria.py"),
    ("2) Create hot samples",         "2_create_hot_samples.py"),
    ("3) Extract important terms",    "3_extracting_important_terms.py"),
    ("4) Crawl top-month (kpop)",     "4_reddit_brand_crawl_kpop.py"),
    ("5) Pivot results",              "5_kpop_sharding2_pivot.py"),
]

LOG_DIR = BASE_DIR.parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LINE = "—" * 72

def run_step(title: str, filename: str) -> None:
    """Run a single step; raise RuntimeError on failure."""
    script_path = (BASE_DIR / filename).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{log_stamp}__{filename}.log"

    print(LINE)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] START {title}")
    print(f"→ Running: {script_path}")
    print(f"→ Logging to: {log_path}")
    t0 = time.time()

    # Run using the same Python interpreter, in BASE_DIR to keep relative paths stable.
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )

    elapsed = time.time() - t0
    hh, rem = divmod(int(elapsed), 3600)
    mm, ss = divmod(rem, 60)

    # Always write full stdout/stderr to log
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"==== STDOUT ({title}) ====\n{proc.stdout}\n")
            f.write(f"\n==== STDERR ({title}) ====\n{proc.stderr}\n")
    except Exception as e:
        print(f"(!) Failed to write log: {e}")

    if proc.returncode != 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FAIL {title} "
              f"(exit={proc.returncode}, elapsed={hh:02d}:{mm:02d}:{ss:02d})")
        print("---- Last lines of STDERR ----")
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        print(tail if tail else "(no stderr)")
        print(f"\nSee full log: {log_path}")
        # Stop the whole pipeline
        raise RuntimeError(f"{title} failed, aborting pipeline.")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE {title} "
              f"(elapsed={hh:02d}:{mm:02d}:{ss:02d}) ✓")

def main():
    print(LINE)
    print("KPOP Pipeline Orchestrator")
    print(f"Base dir: {BASE_DIR}")
    print(f"Logs dir: {LOG_DIR}")
    print(LINE)

    for title, filename in SCRIPTS:
        run_step(title, filename)

    print(LINE)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ALL DONE 🎉")
    print(LINE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(LINE)
        print(f"PIPELINE ABORTED: {e}")
        print(LINE)
        sys.exit(1)