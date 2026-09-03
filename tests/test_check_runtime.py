from scripts.check_runtime import required_runtime_ok, runtime_report


def test_runtime_report_is_json_serializable():
    report = runtime_report()
    assert set(report["modules"]) == {"qfit", "cctbx", "numpy", "scipy", "torch", "gemmi"}
    assert isinstance(required_runtime_ok(report), bool)
