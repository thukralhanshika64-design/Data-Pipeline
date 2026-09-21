"""
Run every project once (smoke test + regenerates all plots and DB rows).

    python run_all.py              # all 10 projects, ~4-6 minutes on one CPU core
    python run_all.py 3 7          # only projects 3 and 7
Logs go to logs/<project>.log
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = sorted(p for p in (ROOT / "projects").iterdir() if p.is_dir() and p.name.startswith("p"))
wanted = {int(a) for a in sys.argv[1:]}
(ROOT / "logs").mkdir(exist_ok=True)

results = []
for p in PROJECTS:
    num = int(p.name[1:3])
    if wanted and num not in wanted:
        continue
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(p / "main.py")], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "logs" / f"{p.name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    results.append((p.name, proc.returncode == 0, time.time() - t0))
    print(f"{'OK  ' if proc.returncode == 0 else 'FAIL'} {p.name:<36} {time.time() - t0:6.1f}s", flush=True)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} projects succeeded.  Logs: {ROOT / 'logs'}")
sys.exit(1 if failed else 0)
