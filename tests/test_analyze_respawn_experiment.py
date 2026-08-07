from scripts.analyze_respawn_experiment import (
    event_mechanism_summary,
    respawn_start_summary,
)


def test_worse_replacement_peak_region_is_reported_as_midpoint():
    check = {
        "site": "SITE",
        "start": "0",
        "step": "100",
        "gram_condition_triggered": "True",
        "rmsd_below_0p3": "False",
        "rmsd_below_0p5": "True",
        "rmsd_below_0p8": "True",
    }
    event = {
        **check,
        "peak_reached_within_0p5_A": "True",
        "endpoint_worse_than_replaced": "True",
        "endpoint_slot_within_1A_of_deposited": "False",
        "endpoint_slot_survived_above_0p10": "False",
        "merged_away_was_near_deposited": "False",
        "endpoint_slot_direct_min_distance_A": "2.5",
        "endpoint_slot_symmetry_min_distance_A": "3.0",
        "endpoint_slot_rotamer_max_deviation_degrees": "10.0",
        "endpoint_slot_canonical": "True",
        "peak_to_A_nearest_atom_distance_A": "1.2",
        "peak_to_B_nearest_atom_distance_A": "1.1",
        "peak_to_midpoint_nearest_atom_distance_A": "0.3",
        "endpoint_missed_state": "B",
        "site_separation_A": "2.0",
        "peak_residual_distance_A": "0.1",
        "peak_in_midpoint_region_within_1A": "True",
    }
    summary = event_mechanism_summary([check], [event], {("SITE", 0)})
    assert summary["endpoint_worse_than_replaced"] == 1
    assert summary["worse_replacement_peak_regions"]["midpoint"] == 1
    assert summary["peak_midpoint_by_site_separation"][
        "below_2p5_A"
    ]["midpoint_region"] == 1


def test_respawn_start_summary_counts_unique_survivors():
    rows = [
        {
            "respawn_event_count": "2",
            "respawned_unique_slot_count": "1",
            "respawned_endpoint_slots_above_0p10": "1",
        },
        {
            "respawn_event_count": "0",
            "respawned_unique_slot_count": "0",
            "respawned_endpoint_slots_above_0p10": "0",
        },
    ]
    summary = respawn_start_summary(rows)
    assert summary["starts_with_respawn"] == 1
    assert summary["total_events"] == 2
    assert summary["total_unique_respawned_slots"] == 1
    assert summary["total_unique_respawned_endpoint_slots_above_0p10"] == 1
