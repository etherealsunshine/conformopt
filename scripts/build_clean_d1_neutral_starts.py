#!/usr/bin/env python3
"""Build and preflight the frozen clean-D1 neutral starts.

This is deliberately start-only.  It never launches the two-slot benchmark,
CV, wider search, or recovery scoring.  The neutral start is the collapsed
occupancy-weighted model refined by A' in its one-slot objective.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from clean_d1_benchmark import collapse_altlocs, site_key, source_path, start_distances
from run_clean_d1_recovery import qfit_sampler
from run_d1_aprime_sequential import APrimeSequential, BACKBONE_NAMES, internal_geometry, rmsd
from run_d1_slot_coordination import FullJointParameterization, build_specs


QUALIFIED = ("8R7O_C_THR1681", "6I3B_B_ALA209", "6ZWK_B_PHE47")
CHAIN_SCOPING = [
    "B-factor arrays: (chain, residue_id, atom_name), with the chain selected before residue lookup",
    "strict seven-residue window extraction: chain-scoped segment and residue IDs",
    "neighbour selection: (chain, residue_number) keys",
    "window sidechain subtraction: (chain, residue_number) keys",
    "deposited-B window and deposited-B B arrays: selected from the requested chain before residue lookup",
    "deposited-A window/truth scoring: selected from the requested chain before residue lookup",
    "qFit input and truth residue lookup: requested chain selected before residue lookup",
]


def _b_vector(structure, window: object) -> np.ndarray:
    """Return a structure's B values in the optimization window's atom order."""
    by_key = {
        (residue.id, name): float(value)
        for chain in structure.chains
        for conformer in chain.conformers
        for segment in conformer.segments
        for residue in segment.residues
        for name, value in zip(residue.name.tolist(), residue.b)
    }
    values = []
    for residue in window.residues:
        for name in residue.name.tolist():
            key = (residue.id, name)
            if key not in by_key:
                raise RuntimeError(f"missing B factor for {key}")
            values.append(by_key[key])
    return np.asarray(values, dtype=float)


def _site_entry(manifest: list[dict[str, object]], key: str) -> dict[str, object]:
    for site in manifest:
        if site_key(site) == key:
            return site
    raise KeyError(f"qualified site {key} is absent from the frozen manifest")


def _initial_axis2_separation(site: dict[str, object], refined_pdb: Path,
                              root: Path, flip_root: Path, device: str) -> dict[str, object]:
    site_tuple = (str(site["pdb_id"]), str(site["chain"]), int(site["resnum"]))
    specs = build_specs(
        root / "nullspace_specs", flip_root, site=site_tuple,
        mask_scope="window", rama_floor=0.02,
        start_pdb=refined_pdb, b_factor_mode="single_conformer",
    )
    selected = next(spec for spec in specs if spec["label"] == "D_null_axis2_30deg")
    p1 = np.zeros(20, dtype=float)
    p2 = np.asarray(selected["p2"], dtype=float)
    parameterization = FullJointParameterization(20)
    import torch
    runner = APrimeSequential(
        root / "nullspace_measurement", 1, 1, *site_tuple,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=device,
        start_pdb=refined_pdb, b_factor_mode="single_conformer",
    )
    packed = parameterization.pack(p1, p2)
    slots = runner.torch_forward(
        parameterization.expand_torch(torch.as_tensor(packed, dtype=torch.float64))
    ).detach().cpu().numpy()
    return {
        "basis": "axis2 of the fixed nullspace SVD direction",
        "angle_deg": 30.0,
        "p2_parameter_norm_deg": float(np.linalg.norm(p2)),
        "p2_parameters_deg": p2.tolist(),
        "slot_1_to_slot_2_backbone_RMSD_A": float(
            rmsd(runner.base.central_backbone(slots[0]), runner.base.central_backbone(slots[1]))
        ),
        "spec_label": selected["label"],
    }


def _mmtbx_alternative(collapsed: Path, refined_root: Path, runner: APrimeSequential,
                       site: dict[str, object]) -> dict[str, object]:
    """Build the available mmtbx real-space/geometry alternative.

    This is intentionally labelled as an alternative route, not as a second
    benchmark start: mmtbx individual-sites refinement is Cartesian and map-
    target based, and does not implement A' seam, Rama/omega, or global-dB
    terms.  The resulting disagreement is therefore diagnostic of the
    cross-tool route, not a prospective comparison result.
    """
    try:
        import iotbx.pdb
        import mmtbx.model
        from mmtbx.refinement.real_space import individual_sites
        from scitbx.array_family import flex
    except Exception as exc:
        return {"available": False, "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        # The pod's CCP4 monomer bundle names the disulfide link ``disulf``
        # while this mmtbx build's geometry code looks it up as ``SS``.
        # Add the compatibility alias only to this in-memory server object;
        # do not edit the shared monomer library or the input structure.
        from mmtbx.monomer_library import server as monomer_server
        original_server = monomer_server.server

        def compatibility_server(*args, **kwargs):
            mon_lib_srv = original_server(*args, **kwargs)
            if ("SS" not in mon_lib_srv.link_link_id_dict
                    and "disulf" in mon_lib_srv.link_link_id_dict):
                mon_lib_srv.link_link_id_dict["SS"] = (
                    mon_lib_srv.link_link_id_dict["disulf"])
            return mon_lib_srv

        monomer_server.server = compatibility_server
        # Keep the independent route scoped to the same strict seven-residue
        # window as the benchmark.  Parsing the entire deposited model can
        # fail on unrelated modified residues elsewhere in the structure
        # (for example 8R7O contains unknown nonbonded energy symbols outside
        # this window).  The window itself is still the complete input model
        # for this cross-check, with all of its atoms selected.
        window_ids = {
            (int(residue.id[0]), str(residue.id[1]).strip())
            for residue in runner.window.residues
        }
        window_pdb = refined_root / "mmtbx_window_input.pdb"
        filtered_lines = []
        for line in collapsed.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                key = (
                    line[21].strip(),
                    (int(line[22:26]), line[26].strip()),
                )
                if key[1] not in window_ids or key[0] != str(site["chain"]):
                    continue
            elif not line.startswith(("CRYST1", "MODEL", "ENDMDL", "TER")):
                continue
            filtered_lines.append(line)
        window_pdb.write_text("\n".join(filtered_lines) + "\n")
        pdb_input = iotbx.pdb.input(file_name=str(window_pdb))
        # A few otherwise standard window atoms in the pod's mmtbx bundle
        # have unknown nonbonded symbols.  Let mmtbx retain them with its
        # available restraints so this diagnostic route can complete; the
        # resulting limitation is recorded in the preflight report.
        model = mmtbx.model.manager(model_input=pdb_input, stop_for_unknowns=False)
        model.process(make_restraints=True)
        shape = tuple(int(value) for value in runner.base.qfit.xmap.array.shape)
        map_data = flex.double(np.asarray(runner.base.qfit.xmap.array, dtype=float).ravel())
        map_data.reshape(flex.grid(shape))
        n_atoms = model.get_xray_structure().scatterers().size()
        selection = flex.bool([True] * n_atoms)
        simple = individual_sites.simple(
            target_map=map_data, selection=selection,
            real_space_gradients_delta=0.10, max_iterations=150,
            geometry_restraints_manager=model.get_restraints_manager(),
        )
        try:
            refined = individual_sites.refinery(
                refiner=simple, xray_structure=model.get_xray_structure(),
                start_trial_weight_value=100.0, rms_bonds_limit=0.02,
                rms_angles_limit=2.0,
            )
            refined_sites = refined.sites_cart_result
            route = "mmtbx.refinement.real_space.individual_sites.simple + refinery"
            refinery_note = None
        except AttributeError as exc:
            # Some mmtbx/CCTBX combinations return a geometry-energy object
            # without the legacy angle_deviations() method used by refinery.
            # Keep the independent real-space route usable with its fixed
            # geometry-restraint weight rather than silently reverting to the
            # A' start.  Do not catch unrelated AttributeErrors.
            if "angle_deviations" not in str(exc):
                raise
            simple.refine(
                xray_structure=model.get_xray_structure().deep_copy_scatterers(),
                weight=100.0,
            )
            refined_sites = simple.sites_cart()
            route = "mmtbx.refinement.real_space.individual_sites.simple (fixed weight fallback)"
            refinery_note = f"refinery unavailable: {type(exc).__name__}: {exc}"
        model.set_sites_cart(refined_sites)
        alternative_pdb = refined_root / "neutral_start_mmtbx_individual_sites.pdb"
        model.get_hierarchy().write_pdb_file(str(alternative_pdb))
        from clean_d1_benchmark import start_distances
        distances = start_distances(alternative_pdb, site)
        return {
            "available": True, "status": "complete",
            "route": route,
            "model_input_scope": "strict seven-residue window; all window atoms selected",
            "model_input": str(window_pdb),
            "monomer_library_compatibility": "aliased existing disulf link id 'disulf' to mmtbx-requested 'SS' in memory",
            "refinery_compatibility": refinery_note,
            "objective_difference": "Cartesian real-space map target plus mmtbx geometry restraints; no A' seam/Rama/omega/global-dB terms",
            "alternative_pdb": str(alternative_pdb),
            "start_distances": distances,
            "rms_bonds_limit": 0.02, "rms_angles_limit_deg": 2.0,
            "max_iterations": 150, "selected_atom_count": int(selection.count(True)),
        }
    except Exception as exc:
        return {"available": True, "status": "error",
                "route": "mmtbx.refinement.real_space.individual_sites.simple + refinery",
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if "original_server" in locals():
            monomer_server.server = original_server


def build_site(site: dict[str, object], output: Path, flip_root: Path, device: str) -> dict[str, object]:
    key = site_key(site)
    root = output / "sites" / key
    root.mkdir(parents=True, exist_ok=True)
    source, split = source_path(str(site["pdb_id"]))
    collapsed = root / "collapsed_weighted_single_conformer.pdb"
    collapse_record = collapse_altlocs(Path(source), collapsed)
    fit_root = root / "aprime_single_slot_fit"
    fit_root.mkdir(parents=True, exist_ok=True)
    site_tuple = (str(site["pdb_id"]), str(site["chain"]), int(site["resnum"]))
    runner = APrimeSequential(
        fit_root, 80, 6, *site_tuple,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=device,
        start_pdb=collapsed, b_factor_mode="single_conformer",
    )
    refined = root / "neutral_start_aprime_single_slot.pdb"
    cached_result = fit_root / "single_slot_result.json"
    cached_final = fit_root / "single_slot_final.npz"
    if refined.is_file() and cached_result.is_file() and cached_final.is_file():
        fit_result = json.loads(cached_result.read_text())
        cached = np.load(cached_final)
        parameters = np.asarray(cached["parameters"], dtype=float)
        final = {"coordinates": np.asarray(cached["final_window"], dtype=float)}
    else:
        parameters, final_state, fit_result = runner.fit_single_slot()
        final = {"coordinates": final_state["coordinates"]}
    from qfit.structure import Structure
    structure = Structure.fromfile(str(collapsed))
    structure.set_xyz(final["coordinates"], runner.window.selection)
    structure.tofile(str(refined))

    distances = start_distances(refined, site)
    separation = float(distances["A_B_separation_A"])
    avg_to_refined = float(rmsd(runner.initial, final["coordinates"]))
    geometry = internal_geometry(runner.window, runner.initial, final["coordinates"])

    truth_a = runner.base.truth_a_structure
    truth_b = runner.base.b_structure
    deposited_a_b = _b_vector(truth_a, runner.window)[runner.base.model_atom_indices]
    deposited_b_b = _b_vector(truth_b, runner.window)[runner.base.model_atom_indices]
    weighted_input_b = np.asarray(runner.base.b_factors_a_model, dtype=float)
    effective_start_b = weighted_input_b + float(fit_result["b_offset_A2"])
    slot1_b = runner.base.slot_b_factors(0, float(fit_result["b_offset_A2"]))
    slot2_b = runner.base.slot_b_factors(1, float(fit_result["b_offset_A2"]))
    b_arrays = {
        "array_order": "seven-residue window backbone atoms in renderer model-atom order",
        "deposited_A": deposited_a_b.tolist(),
        "deposited_B": deposited_b_b.tolist(),
        "occupancy_weighted_collapsed_input": weighted_input_b.tolist(),
        "fitted_global_dB_A2": float(fit_result["b_offset_A2"]),
        "effective_single_conformer_start": effective_start_b.tolist(),
        "renderer_slot_1": slot1_b.tolist(),
        "renderer_slot_2": slot2_b.tolist(),
        "renderer_slot_arrays_identical": bool(np.array_equal(slot1_b, slot2_b)),
        "start_mean_vs_deposited_A": float(np.mean(weighted_input_b - deposited_a_b)),
        "start_mean_vs_deposited_B": float(np.mean(weighted_input_b - deposited_b_b)),
    }

    sidechain = runner.base.start_sidechain_subtraction_mismatch()
    axis2 = _initial_axis2_separation(site, refined, root, flip_root, device)
    qfit = qfit_sampler(site, refined)
    candidate0 = qfit["candidates"][0]

    mmtbx = {"module": "mmtbx.refinement", "available": False}
    try:
        importlib.import_module("mmtbx.refinement")
    except Exception as exc:  # availability is an environment fact, not a site failure
        mmtbx["error"] = f"{type(exc).__name__}: {exc}"
    else:
        mmtbx = _mmtbx_alternative(collapsed, root, runner, site)

    row = {
        "site": key, "pdb_id": site["pdb_id"], "chain": site["chain"],
        "resnum": int(site["resnum"]), "source_split": split,
        "procedure": {
            "step_1": "occupancy-weight-average A/B altloc-group coordinates and B factors; blank altloc; set occupancy 1.00",
            "step_2": "A' single-slot fit: 20 torsions, free occupancy, same window mask/renderer/seam/Rama/omega, fitted global dB",
            "step_3": "all renderer-mask voxels used for the start fit; no CV or held-out voxels",
            "phenix": "not used",
            "collapsed_input": str(collapsed), "refined_start": str(refined),
            "collapse_record": collapse_record,
        },
        "fit": fit_result,
        "average_input_to_refined_output_window_RMSD_A": avg_to_refined,
        "average_input_to_refined_output_geometry": geometry,
        "start_distances": {
            **distances,
            "start_to_A_fraction_of_AB": float(distances["start_rmsd_to_A_A"] / separation),
            "start_to_B_fraction_of_AB": float(distances["start_rmsd_to_B_A"] / separation),
            "both_clear_15_percent_floor": bool(
                distances["start_rmsd_to_A_A"] >= 0.15 * separation
                and distances["start_rmsd_to_B_A"] >= 0.15 * separation
            ),
        },
        "b_factor_arrays": b_arrays,
        "sidechain_subtraction_residual": sidechain,
        "initialisation": axis2,
        "qfit": {
            "candidate_0": candidate0,
            "candidate_count": qfit["candidate_count"],
            "b_factor_provenance": qfit["b_factor_provenance"],
            "candidate_0_b_factors": qfit.get("candidate_0_b_factors"),
            "candidate_0_b_equals_input_residue": qfit.get("candidate_0_b_equals_input_residue"),
        },
        "mmtbx_alternative": mmtbx,
    }
    (root / "preflight.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flip-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--site", default=None,
                        help="build only this frozen site key")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    # The detached launcher creates the log directory before starting us.
    # Keep that checkpointing setup intact and never delete or overwrite an
    # existing site tree.
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_keys = QUALIFIED if args.site is None else (args.site,)
    unknown = sorted(set(selected_keys) - set(QUALIFIED))
    if unknown:
        raise ValueError(f"site is not in the frozen qualified set: {unknown}")
    for key in selected_keys:
        rows.append(build_site(_site_entry(manifest, key), args.output, args.flip_root, args.device))
        (args.output / "progress.json").write_text(json.dumps({
            "status": "running", "completed": len(rows), "total": len(QUALIFIED),
        }, indent=2))
    report = {
        "status": "complete", "benchmark_started": False,
        "wider_search_started": False, "neutral_start_only": True,
        "selected_site_keys": list(selected_keys),
        "sites": rows,
        "fixed_parameters": {
            "rama_floor": 0.02, "omega_restraint_scale_deg": 5.0,
            "planar_weight": 0.05, "rho": 2.0 / (1.6275900803874028 ** 2),
            "MIQP_K": 4, "t_min": 0.02, "success_threshold_fraction": 0.30,
            "nullspace_basis": "axis2", "nullspace_angle_deg": 30.0,
        },
        "chain_scoped_lookups": CHAIN_SCOPING,
        "heldout_guard": {
            "status": "not_exercised_in_start_only_preflight",
            "code_guard": "assert_heldout_geometry_provenance requires fit_scope == 'training-only', exact training fit_voxel_indices, total mask cardinality, and matching voxel-index SHA256",
        },
    }
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (args.output / "progress.json").write_text(json.dumps({
        "status": "complete", "completed": len(rows), "total": len(QUALIFIED),
    }, indent=2))


if __name__ == "__main__":
    main()
