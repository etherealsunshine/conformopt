from scripts.d1_population_calibrated_weights import (
    D1_OMEGA_SCALE_DEG,
    D1_RAMA_FLOOR,
    D1_WEIGHT_CALIBRATION_SOURCE_ROWS,
    d1_weight_provenance,
)


def test_d1_defaults_are_population_calibrated_and_provenanced():
    assert D1_RAMA_FLOOR == 0.0016931321476603015
    assert D1_OMEGA_SCALE_DEG == 13.019150002471157
    provenance = d1_weight_provenance()
    assert provenance["source_rows"] == D1_WEIGHT_CALIBRATION_SOURCE_ROWS == 232_890
    assert provenance["rama_floor"] == D1_RAMA_FLOOR
    assert provenance["omega_scale_deg"] == D1_OMEGA_SCALE_DEG
