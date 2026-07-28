import math

import gemmi
import numpy as np
import torch

from density_denoiser.audit_five_site_endpoints import (
    merge_then_assign_conformers,
    optimal_ab_assignments,
    write_tmol_segment,
)
from density_denoiser.clash_environment import (
    EnvironmentAtom,
    SoftEnvironmentRecord,
    compatible_spatial_metrics,
    partition_soft_environment,
    soft_clash_penalty,
    soft_clash_barrier_penalty,
)
from density_denoiser.summarize_endpoint_audit import (
    matched_tmol_evaluation,
    select_assigned_pair,
)


def test_tmol_uses_matched_assignment_not_better_control():
    energy = {"tmol_energy": "577.0", "tmol_A": "577.7", "tmol_B": "358.6"}
    reference, delta, valid = matched_tmol_evaluation(
        {"assignment": "A"}, energy
    )
    assert reference == 577.7
    assert math.isclose(delta, -0.7)
    assert valid

    reference, delta, valid = matched_tmol_evaluation(
        {"assignment": "B"}, energy
    )
    assert reference == 358.6
    assert delta > 200
    assert not valid


def test_tmol_rejects_unmatched_active_conformer():
    reference, delta, valid = matched_tmol_evaluation(
        {"assignment": "other"},
        {"tmol_energy": "1", "tmol_A": "2", "tmol_B": "3"},
    )
    assert math.isnan(reference)
    assert math.isnan(delta)
    assert not valid


def test_assigned_pair_selects_best_found_A_and_B_and_ignores_extras():
    rows = [
        {
            "candidate_id": "A-low-occ",
            "assignment": "A",
            "occupancy": "0.08",
            "rmsd_to_A_conventional": "0.01",
        },
        {
            "candidate_id": "A-found",
            "assignment": "A",
            "occupancy": "0.30",
            "rmsd_to_A_conventional": "0.20",
        },
        {
            "candidate_id": "B-found",
            "assignment": "B",
            "occupancy": "0.40",
            "rmsd_to_B_conventional": "0.15",
        },
        {
            "candidate_id": "other",
            "assignment": "other",
            "occupancy": "0.22",
            "rmsd_to_A_conventional": "2.0",
        },
    ]
    pair = select_assigned_pair(rows)
    assert pair is not None
    assert pair["A"]["candidate_id"] == "A-found"
    assert pair["B"]["candidate_id"] == "B-found"


def test_optimal_assignment_recovers_overlapping_state_with_distinct_slots():
    assignments = optimal_ab_assignments(
        occupancies=[0.40, 0.35, 0.20, 0.05],
        rmsd_a=[0.10, 0.30, 2.00, 0.05],
        rmsd_b=[0.20, 2.00, 2.00, 0.05],
        active_occupancy=0.05,
        found_occupancy=0.10,
        rmsd_cutoff=1.0,
    )
    assert assignments == ["B", "A", "other", "inactive"]


def test_optimal_assignment_respects_cutoff_and_reporting_occupancy():
    assignments = optimal_ab_assignments(
        occupancies=[0.09, 0.40, 0.30],
        rmsd_a=[0.01, 1.00, 2.00],
        rmsd_b=[2.00, 2.00, 0.20],
        active_occupancy=0.05,
        found_occupancy=0.10,
        rmsd_cutoff=1.0,
    )
    # The sub-reporting-threshold slot cannot claim A, and exactly 1.0 A does
    # not satisfy the unchanged strict "< 1.0 A" recovery definition.
    assert assignments == ["other", "other", "B"]


def test_merge_then_assign_sums_duplicate_occupancy_before_matching():
    result = merge_then_assign_conformers(
        occupancies=[0.28, 0.29, 0.40, 0.03],
        rmsd_a=[0.20, 0.10, 2.00, 0.05],
        rmsd_b=[2.00, 2.00, 0.15, 0.05],
        pairwise_rmsd=[
            [0.00, 0.20, 2.00, 1.00],
            [0.20, 0.00, 2.00, 1.00],
            [2.00, 2.00, 0.00, 1.00],
            [1.00, 1.00, 1.00, 0.00],
        ],
        active_occupancy=0.05,
        found_occupancy=0.10,
        rmsd_cutoff=1.0,
        merge_rmsd_threshold=0.30,
    )
    assert result["clusters"] == [[0, 1], [2]]
    assert math.isclose(result["cluster_occupancies"][0], 0.57)
    assert math.isclose(result["cluster_occupancies"][1], 0.40)
    assert result["representatives"] == [1, 2]
    assert result["assignments"] == ["merged_duplicate", "A", "B", "inactive"]
    assert math.isclose(result["cluster_occupancy"][1], 0.57)


def test_merge_then_assign_can_activate_a_summed_subthreshold_cluster():
    result = merge_then_assign_conformers(
        occupancies=[0.06, 0.06, 0.88],
        rmsd_a=[0.10, 0.20, 2.00],
        rmsd_b=[2.00, 2.00, 0.10],
        pairwise_rmsd=[
            [0.00, 0.10, 2.00],
            [0.10, 0.00, 2.00],
            [2.00, 2.00, 0.00],
        ],
        active_occupancy=0.05,
        found_occupancy=0.10,
        rmsd_cutoff=1.0,
        merge_rmsd_threshold=0.30,
    )
    assert result["cluster_occupancies"] == [0.12, 0.88]
    assert sorted(result["assignments"]) == ["A", "B", "merged_duplicate"]


def test_merge_protects_distinct_ab_anchors_inside_distance_threshold():
    result = merge_then_assign_conformers(
        occupancies=[0.45, 0.45, 0.10],
        rmsd_a=[0.10, 0.30, 2.00],
        rmsd_b=[0.30, 0.10, 2.00],
        pairwise_rmsd=[
            [0.00, 0.20, 2.00],
            [0.20, 0.00, 2.00],
            [2.00, 2.00, 0.00],
        ],
        active_occupancy=0.05,
        found_occupancy=0.10,
        rmsd_cutoff=1.0,
        merge_rmsd_threshold=0.30,
    )
    assert result["clusters"] == [[0], [1], [2]]
    assert result["assignments"][:2] == ["A", "B"]


def test_hard_clash_uses_best_neighbor_altloc_state():
    candidate = np.asarray([[0.0, 0.0, 0.0]])
    atoms = [
        EnvironmentAtom(
            xyz=(1.5, 0.0, 0.0), label="LYS95:B", residue_group="B:95",
            altloc="B", occupancy=0.68, is_water=False,
        ),
        EnvironmentAtom(
            xyz=(3.0, 0.0, 0.0), label="LYS95:C", residue_group="B:95",
            altloc="C", occupancy=0.15, is_water=False,
        ),
    ]
    result = compatible_spatial_metrics(candidate, atoms, 2.0, "A")
    assert result["closest_atom"] == "LYS95:C"
    assert result["minimum_distance"] == 3.0
    assert result["no_clash"]


def test_water_labels_and_occupancy_scale_hard_cutoff():
    candidate = np.asarray([[0.0, 0.0, 0.0]])
    labeled = EnvironmentAtom(
        xyz=(0.6, 0.0, 0.0), label="HOH186:B", residue_group="water:186",
        altloc="B", occupancy=0.67, is_water=True,
    )
    assert compatible_spatial_metrics(
        candidate, [labeled], 2.0, "A"
    )["no_clash"]
    assert not compatible_spatial_metrics(
        candidate, [labeled], 2.0, "B"
    )["no_clash"]

    partial_unlabeled = EnvironmentAtom(
        xyz=(1.5, 0.0, 0.0), label="HOH58", residue_group="water:58",
        altloc="", occupancy=0.65, is_water=True,
    )
    result = compatible_spatial_metrics(
        candidate, [partial_unlabeled], 2.0, "A"
    )
    assert result["minimum_distance"] == 1.5
    assert result["no_clash"]  # effective cutoff is 2.0 * 0.65 = 1.3 A.


def test_soft_environment_gives_partial_labeled_water_an_absent_state():
    records = [
        SoftEnvironmentRecord(
            xyz=(0.6, 0.0, 0.0),
            group_key="water:186",
            atom_name="O",
            altloc="B",
            occupancy=0.67,
            is_water=True,
        )
    ]
    invariant, weights, alternate, invariant_records = (
        partition_soft_environment(records, torch.device("cpu"))
    )
    assert not invariant.numel()
    assert not weights.numel()
    assert not invariant_records
    assert len(alternate) == 1
    assert any(not state.numel() for state in alternate[0])
    loss = soft_clash_penalty(
        torch.tensor([[0.0, 0.0, 0.0]]),
        invariant,
        weights,
        alternate,
        threshold=3.0,
    )
    assert torch.isclose(loss, torch.tensor(0.0))


def test_soft_environment_requires_clearance_across_full_water_states():
    records = [
        SoftEnvironmentRecord(
            xyz=(1.0, 0.0, 0.0),
            group_key="water:10",
            atom_name="O",
            altloc="A",
            occupancy=0.50,
            is_water=True,
        ),
        SoftEnvironmentRecord(
            xyz=(2.0, 0.0, 0.0),
            group_key="water:10",
            atom_name="O",
            altloc="B",
            occupancy=0.50,
            is_water=True,
        ),
    ]
    invariant, weights, alternate, _records = partition_soft_environment(
        records, torch.device("cpu")
    )
    assert all(state.numel() for state in alternate[0])
    loss = soft_clash_penalty(
        torch.tensor([[0.0, 0.0, 0.0]]),
        invariant,
        weights,
        alternate,
        threshold=3.0,
    )
    assert torch.isclose(loss, torch.tensor(1.0))


def test_symmetry_barrier_is_expensive_at_the_hard_gate():
    candidate = torch.tensor([[0.0, 0.0, 0.0]])
    environment = torch.tensor([[2.0, 0.0, 0.0]])
    loss = soft_clash_barrier_penalty(
        candidate,
        environment,
        torch.ones(1),
        [],
        soft_threshold=2.5,
        hard_threshold=2.0,
        barrier_buffer=0.25,
        barrier_scale=1.0,
    )
    assert torch.isclose(loss, torch.tensor(1.25))


def test_zero_scale_symmetry_barrier_reproduces_squared_hinge():
    candidate = torch.tensor([[0.0, 0.0, 0.0]])
    environment = torch.tensor([[1.8, 0.0, 0.0]])
    loss = soft_clash_barrier_penalty(
        candidate,
        environment,
        torch.ones(1),
        [],
        soft_threshold=2.5,
        hard_threshold=2.0,
        barrier_buffer=0.25,
        barrier_scale=0.0,
    )
    assert torch.isclose(loss, torch.tensor(0.49), atol=1e-6)


def test_tmol_environment_selects_alt_state_without_blank_atom_tie(tmp_path):
    pdb = """\
ATOM      1  N   SER A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  SER A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   SER A   1      12.000  10.000  10.000  1.00 20.00           C
ATOM      4  O   SER A   1      13.000  10.000  10.000  1.00 20.00           O
ATOM      5  CB ASER A   1       0.000   0.000   0.000  0.50 20.00           C
ATOM      6  OG ASER A   1       0.000   1.000   0.000  0.50 20.00           O
ATOM      7  CB BSER A   1       0.000   0.000   0.000  0.50 20.00           C
ATOM      8  OG BSER A   1       0.000  -1.000   0.000  0.50 20.00           O
ATOM      9  N   SER A   2       0.100   0.000   0.000  1.00 20.00           N
ATOM     10  CA  SER A   2       5.000   5.000   5.000  1.00 20.00           C
ATOM     11  C   SER A   2       6.000   5.000   5.000  1.00 20.00           C
ATOM     12  O   SER A   2       7.000   5.000   5.000  1.00 20.00           O
ATOM     13  CB  SER A   2       8.000   5.000   5.000  1.00 20.00           C
ATOM     14  OG ASER A   2       1.000   0.000   0.000  0.80 20.00           O
ATOM     15  OG BSER A   2       3.000   0.000   0.000  0.20 20.00           O
TER
END
"""
    source = tmp_path / "source.pdb"
    source.write_text(pdb)
    structure = gemmi.read_structure(str(source))
    output = tmp_path / "segment.pdb"
    write_tmol_segment(
        output,
        structure,
        "A",
        1,
        environment_coordinates=np.asarray([[0.0, 0.0, 0.0]]),
        target_altloc="A",
    )
    neighbor_og = next(
        line for line in output.read_text().splitlines()
        if line.startswith("ATOM")
        and int(line[22:26]) == 2
        and line[12:16].strip() == "OG"
    )
    assert float(neighbor_og[30:38]) == 3.0
