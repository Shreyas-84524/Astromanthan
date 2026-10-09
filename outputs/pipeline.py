"""One-command entry point; reuses Astromanthan's existing ML Model engine."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ML Model'))
from backend.engine.astronomy.workflow import main

if __name__ == '__main__':
    main()
    if len(sys.argv) == 1:
        from audit_proxies import main as audit_proxies
        audit_proxies()
        from verify_outputs import main as verify_outputs
        verify_outputs()
        from refresh_report import main as refresh_report
        refresh_report()
