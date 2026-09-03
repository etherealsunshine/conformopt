#!/usr/bin/env python3
"""Report the external runtime pieces used by the qFit-to-A-prime pipeline."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil


def _module_status(name: str, distribution: str | None = None) -> dict[str, object]:
    spec = importlib.util.find_spec(name)
    import_error = None
    imported = False
    module = None
    if spec is not None:
        try:
            module = importlib.import_module(name)
            imported = True
        except Exception as exc:  # report broken native stacks without aborting the check
            import_error = f"{type(exc).__name__}: {exc}"
    version = None
    if distribution is not None:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "available": spec is not None,
        "importable": imported,
        "version": version,
        "location": str(spec.origin) if spec is not None and spec.origin else None,
        "import_error": import_error,
        **({"cuda_available": bool(module.cuda.is_available())}
           if name == "torch" and module is not None and hasattr(module, "cuda") else {}),
    }


def runtime_report() -> dict[str, object]:
    """Return a JSON-serializable report with import and executable checks."""
    phenix_root = os.environ.get("PHENIX_ROOT") or os.environ.get("PHENIX")
    return {
        "modules": {
            "qfit": _module_status("qfit"),
            "cctbx": _module_status("cctbx"),
            "numpy": _module_status("numpy", "numpy"),
            "scipy": _module_status("scipy", "scipy"),
            "torch": _module_status("torch", "torch"),
            "gemmi": _module_status("gemmi", "gemmi"),
        },
        "executables": {
            "phenix.refine": shutil.which("phenix.refine"),
        },
        "phenix_root": phenix_root,
        "python": os.sys.executable,
    }


def required_runtime_ok(report: dict[str, object], *, clash_weight: float = 0.0) -> bool:
    modules = report["modules"]
    required = ("qfit", "cctbx", "numpy", "scipy", "torch")
    if not all(bool(modules[name]["importable"]) for name in required):
        return False
    if clash_weight > 0.0 and not bool(modules["gemmi"]["importable"]):
        return False
    if clash_weight > 0.0 and not report.get("phenix_root"):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero when required runtime modules are unavailable")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    report = runtime_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, record in report["modules"].items():
            state = "importable" if record["importable"] else (
                "discoverable" if record["available"] else "missing"
            )
            version = f" ({record['version']})" if record["version"] else ""
            print(f"{name}: {state}{version}")
            if record["import_error"]:
                print(f"  import_error: {record['import_error']}")
        for name, executable in report["executables"].items():
            print(f"{name}: {executable or 'missing'}")
        print(f"PHENIX_ROOT: {report['phenix_root'] or 'unset'}")
        print(f"python: {report['python']}")
    return 0 if not args.strict or required_runtime_ok(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
