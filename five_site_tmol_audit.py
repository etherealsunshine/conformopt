"""Compatibility entry point for the active prospective controller.

New commands should invoke ``scripts/five_site_tmol_audit.py``.  This shim must
remain until the run launched on 2026-07-21 has completed its tmol stage.
"""

from scripts.five_site_tmol_audit import main


if __name__ == "__main__":
    main()
