import math

import torch

from density_denoiser.residue_geometry import (
    CHI_SPECS,
    canonical_centers_radians,
    canonical_width_degrees,
    classify_rotamer_degrees,
    reference_permutations,
    symmetry_aware_rmsd,
)


EXPECTED_CHI_COUNTS = {
    "ARG": 4, "ASN": 2, "ASP": 2, "CYS": 1, "GLN": 3, "GLU": 3,
    "HIS": 2, "ILE": 2, "LEU": 2, "LYS": 4, "MET": 3, "PHE": 2,
    "SER": 1, "THR": 1, "TRP": 2, "TYR": 2, "VAL": 1,
}


def test_production_topology_covers_all_nonproline_chi_residues():
    assert {name: len(spec["dihedrals"]) for name, spec in CHI_SPECS.items()} == (
        EXPECTED_CHI_COUNTS
    )
    for spec in CHI_SPECS.values():
        assert len(spec["dihedrals"]) == len(spec["rotations"])
        for quartet, (origin, endpoint, downstream) in zip(
            spec["dihedrals"], spec["rotations"]
        ):
            assert len(quartet) == 4
            assert origin == quartet[1]
            assert endpoint == quartet[2]
            assert quartet[3] in downstream
            assert origin not in downstream
            assert endpoint not in downstream


def test_residue_specific_terminal_centers():
    planar = (0.0, math.pi, -math.pi)
    for residue, index in (("ASN", 1), ("ASP", 1), ("GLN", 2), ("GLU", 2)):
        assert canonical_centers_radians(residue, index) == planar
        assert canonical_width_degrees(residue, index) == 90.0
    assert canonical_centers_radians("ARG", 3) == (
        -math.pi / 2, math.pi / 2, math.pi, -math.pi
    )
    assert canonical_centers_radians("TRP", 1) == (
        0.0, math.radians(-105.0), math.radians(105.0)
    )
    assert canonical_centers_radians("HIS", 1) == (
        math.radians(-170.0), math.radians(-80.0),
        math.radians(80.0), math.radians(170.0),
    )
    for residue in ("PHE", "TYR"):
        assert canonical_centers_radians(residue, 1) == (
            -math.pi / 2, math.pi / 2
        )
        assert canonical_width_degrees(residue, 1) == 45.0
    assert canonical_centers_radians("MET", 2) == (
        -math.pi / 2, math.pi / 2, math.pi, -math.pi
    )


def test_rotamer_gate_uses_per_chi_widths():
    _state, deviations, widths, valid = classify_rotamer_degrees(
        "ASN", [18.0, 81.7]
    )
    assert math.isclose(deviations[0], 42.0)
    assert math.isclose(deviations[1], 81.7)
    assert widths == [45.0, 90.0]
    assert valid

    _state, deviations, widths, valid = classify_rotamer_degrees(
        "TRP", [-99.9, 8.3]
    )
    assert widths == [45.0, 60.0]
    assert math.isclose(deviations[1], 8.3)
    assert valid

    for his_chi2 in (-170.0, -80.0, 80.0, 170.0):
        _state, deviations, widths, valid = classify_rotamer_degrees(
            "HIS", [-60.0, his_chi2]
        )
        assert deviations == [0.0, 0.0]
        assert widths == [45.0, 60.0]
        assert valid


def test_symmetry_aware_rmsd_accepts_equivalent_terminal_labels():
    names = ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    reference = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    permutation = reference_permutations(names, "PHE")[1]
    candidate = reference[permutation]
    raw = torch.sqrt(torch.mean(torch.sum((candidate - reference).square(), dim=-1)))
    assert raw > 0
    assert symmetry_aware_rmsd(candidate, reference, names, "PHE") == 0


def test_aromatic_equivalence_requires_coupled_ring_swap():
    names = ["CD1", "CD2", "CE1", "CE2"]
    assert reference_permutations(names, "PHE") == [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
    ]
