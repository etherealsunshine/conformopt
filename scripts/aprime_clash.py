"""Differentiable non-bonded contact residuals for the A-prime panel.

The optimizer works on a fixed seven-residue window.  This module builds a
static list of dynamic-window/environment and within-slot window contacts
from the deposited A structure, removes graph-distance 1-2 and 1-3 pairs,
and exposes the remaining overlap penalties as least-squares residual rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


@dataclass(frozen=True)
class ClashResidualContext:
    """Static contact topology and thresholds for one optimization window."""

    dynamic_pair_i: torch.Tensor
    dynamic_pair_j: torch.Tensor
    dynamic_pair_threshold: torch.Tensor
    environment_pair_i: torch.Tensor
    environment_xyz: torch.Tensor
    environment_threshold: torch.Tensor
    pair_cutoff_A: float
    threshold_scale: float
    dynamic_atom_count: int
    source_pdb: str

    @property
    def pair_count(self) -> int:
        return int(self.dynamic_pair_i.numel() + self.environment_pair_i.numel())

    @property
    def residual_pair_count(self) -> int:
        """Number of residual rows emitted for the two optimized slots."""
        return 2 * self.pair_count

    def residual(
        self,
        coordinates: torch.Tensor,
        weight: float,
        *,
        normalize_by_pair_count: bool = False,
    ) -> torch.Tensor:
        """Return weighted overlap residuals for both slots.

        When requested, normalize by the number of emitted slot-pair rows.
        This makes the calibrated coefficient refer to the aggregate clash
        energy for a typical overlap population rather than to the raw number
        of monitored contacts in a particular window.
        """
        if coordinates.ndim != 3 or coordinates.shape[0] != 2:
            raise ValueError("coordinates must have shape (2, atoms, 3)")
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("clash weight must be finite and non-negative")
        residual_weight = float(weight)
        if normalize_by_pair_count:
            residual_weight /= max(self.residual_pair_count, 1)
        zero = coordinates.new_empty((0,))
        rows = []
        if self.dynamic_pair_i.numel():
            i = self.dynamic_pair_i.to(coordinates.device)
            j = self.dynamic_pair_j.to(coordinates.device)
            threshold = self.dynamic_pair_threshold.to(coordinates.device)
            distance = torch.linalg.vector_norm(coordinates[:, i] - coordinates[:, j], dim=-1)
            rows.append(torch.relu(threshold[None, :] - distance).reshape(-1))
        if self.environment_pair_i.numel():
            i = self.environment_pair_i.to(coordinates.device)
            environment = self.environment_xyz.to(coordinates.device)
            threshold = self.environment_threshold.to(coordinates.device)
            distance = torch.linalg.vector_norm(coordinates[:, i] - environment[None, :, :], dim=-1)
            rows.append(torch.relu(threshold[None, :] - distance).reshape(-1))
        if not rows:
            return zero
        return np.sqrt(residual_weight) * torch.cat(rows)

    def distances(self, coordinates: np.ndarray) -> np.ndarray:
        """Return all monitored distances, useful for diagnostics/tests."""
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (2, self.dynamic_atom_count, 3):
            raise ValueError("coordinates have the wrong shape")
        rows = []
        if self.dynamic_pair_i.numel():
            i = self.dynamic_pair_i.numpy()
            j = self.dynamic_pair_j.numpy()
            rows.append(np.linalg.norm(coordinates[:, i] - coordinates[:, j], axis=-1).reshape(-1))
        if self.environment_pair_i.numel():
            i = self.environment_pair_i.numpy()
            xyz = self.environment_xyz.numpy()
            rows.append(np.linalg.norm(coordinates[:, i] - xyz[None, :, :], axis=-1).reshape(-1))
        return np.concatenate(rows) if rows else np.empty(0, dtype=float)


def _vdw_radius(element: str) -> float:
    # qFit's default ClashDetector uses the corresponding element radii.  The
    # panel is heavy-atom-only, and these values are the standard qFit/Probe
    # radii for the elements present in the protein test set.
    radii = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
             "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
    key = str(element).strip().upper()
    if key not in radii:
        raise ValueError(f"no heavy-atom van der Waals radius for {element!r}")
    return radii[key]


def _primary_atoms(path: Path) -> list[dict]:
    from analyze_phenix_clash_audit import read_pdb_atoms

    chosen: dict[tuple[str, str, str, str, str], dict] = {}
    for atom in read_pdb_atoms(path):
        if atom["element"].startswith(("H", "D")):
            continue
        if atom["altloc"] not in ("", "A"):
            continue
        key = (atom["chain"], atom["resseq"], atom["icode"], atom["resname"], atom["name"])
        # Prefer A over blank when both records exist; blank atoms are still
        # retained for invariant atoms and for structures without altloc A.
        if key not in chosen or atom["altloc"] == "A":
            chosen[key] = atom
    return list(chosen.values())


def build_context_for_runner(
    runner,
    *,
    pair_cutoff_A: float = 4.5,
    threshold_scale: float = 0.75,
    monomer_root: Path | None = None,
) -> ClashResidualContext:
    """Build contacts for a seven-residue A-prime runner.

    Dynamic atoms are every heavy atom in the runner window.  The fixed
    environment is the deposited A conformer outside that window.  Contacts
    are selected at the deposited-A coordinates and use the same CCP4
    monomer-library graph filter as the downstream clash audit.
    """
    if pair_cutoff_A <= 0.0 or threshold_scale <= 0.0:
        raise ValueError("pair cutoff and threshold scale must be positive")
    from analyze_phenix_clash_audit import Connectivity
    from scipy.spatial import cKDTree

    source = Path(runner.base.truth_path)
    if monomer_root is None:
        phenix_root = Path("/home/dev/qfit_unet_data/phenix-2.2-6143")
        candidates = (
            phenix_root / "lib" / "python3.11" / "site-packages" / "chem_data" / "chemical_components",
            phenix_root / "lib" / "python3.11" / "site-packages" / "mmtbx" / "chemical_components",
            phenix_root / "chem_data" / "chemical_components",
        )
        monomer_root = next((path for path in candidates if path.is_dir()), candidates[0])
    connectivity = Connectivity(source, monomer_root)
    atoms = _primary_atoms(source)
    window_residues = {
        (runner.base.chain, str(int(residue.id[0])), str(residue.id[1]).strip(), str(residue.resn[0]).upper())
        for residue in runner.window.residues
    }
    dynamic_keys = []
    dynamic_elements = []
    dynamic_coords = []
    dynamic_local_indices = []
    for local, (name, element, chain, resi, icode, resname) in enumerate(zip(
            runner.window.name, runner.window.e, runner.window.chain,
            runner.window.resi, runner.window.icode, runner.window.resn)):
        if str(element).strip().upper().startswith(("H", "D")):
            continue
        dynamic_keys.append((str(chain).strip(), str(int(resi)), str(icode).strip(),
                             str(resname).strip().upper(), str(name).strip().upper()))
        dynamic_elements.append(str(element).strip().upper())
        dynamic_coords.append(runner.initial[local])
        dynamic_local_indices.append(local)
    if not dynamic_keys:
        raise ValueError("clash context has no heavy dynamic atoms")
    dynamic_xyz = np.asarray(dynamic_coords, dtype=float)
    dynamic_records = []
    for key, element, xyz in zip(dynamic_keys, dynamic_elements, dynamic_xyz):
        dynamic_records.append({
            "chain": key[0], "resseq": key[1], "icode": key[2], "resname": key[3],
            "name": key[4], "element": element, "xyz": tuple(xyz),
        })
    environment = [
        atom for atom in atoms
        if (atom["chain"], atom["resseq"], atom["icode"], atom["resname"]) not in window_residues
    ]
    env_xyz = np.asarray([atom["xyz"] for atom in environment], dtype=float)
    env_tree = cKDTree(env_xyz) if len(environment) else None
    dynamic_tree = cKDTree(dynamic_xyz)
    dynamic_pairs: list[tuple[int, int, float]] = []
    for i, j in sorted(dynamic_tree.query_pairs(pair_cutoff_A)):
        a, b = dynamic_records[i], dynamic_records[j]
        if connectivity.distance(a, b, 2) is not None:
            continue
        dynamic_pairs.append((dynamic_local_indices[i], dynamic_local_indices[j],
                              threshold_scale * (_vdw_radius(a["element"]) + _vdw_radius(b["element"]))))
    environment_pairs: list[tuple[int, np.ndarray, float]] = []
    if env_tree is not None:
        for i, xyz in enumerate(dynamic_xyz):
            for env_index in env_tree.query_ball_point(xyz, pair_cutoff_A):
                a, b = dynamic_records[i], environment[env_index]
                if connectivity.distance(a, b, 2) is not None:
                    continue
                threshold = threshold_scale * (_vdw_radius(a["element"]) + _vdw_radius(b["element"]))
                environment_pairs.append((dynamic_local_indices[i], environment[env_index]["xyz"], threshold))
    return ClashResidualContext(
        dynamic_pair_i=torch.tensor([x[0] for x in dynamic_pairs], dtype=torch.long),
        dynamic_pair_j=torch.tensor([x[1] for x in dynamic_pairs], dtype=torch.long),
        dynamic_pair_threshold=torch.tensor([x[2] for x in dynamic_pairs], dtype=torch.float64),
        environment_pair_i=torch.tensor([x[0] for x in environment_pairs], dtype=torch.long),
        environment_xyz=torch.tensor([x[1] for x in environment_pairs], dtype=torch.float64).reshape(-1, 3),
        environment_threshold=torch.tensor([x[2] for x in environment_pairs], dtype=torch.float64),
        pair_cutoff_A=float(pair_cutoff_A), threshold_scale=float(threshold_scale),
        dynamic_atom_count=len(runner.initial), source_pdb=str(source),
    )


def population_contact_statistics(
    pdb_paths: list[Path], monomer_root: Path, *,
    pair_cutoff_A: float = 4.5, threshold_scale: float = 0.75,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Collect deposited primary-state non-bonded distances and overlaps."""
    from analyze_phenix_clash_audit import Connectivity
    from scipy.spatial import cKDTree

    distances: list[float] = []
    overlaps: list[float] = []
    structures_with_contacts = 0
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "status": "running", "processed_structures": 0,
            "total_structures": len(pdb_paths),
        }, indent=2) + "\n")
    for processed, path in enumerate(pdb_paths, start=1):
        atoms = _primary_atoms(path)
        if not atoms:
            continue
        connectivity = Connectivity(path, monomer_root)
        xyz = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
        tree = cKDTree(xyz)
        structure_distances = []
        for i, j in sorted(tree.query_pairs(pair_cutoff_A)):
            if connectivity.distance(atoms[i], atoms[j], 2) is not None:
                continue
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            threshold = threshold_scale * (_vdw_radius(atoms[i]["element"]) + _vdw_radius(atoms[j]["element"]))
            distances.append(distance)
            overlaps.append(max(0.0, threshold - distance))
            structure_distances.append(distance)
        if structure_distances:
            structures_with_contacts += 1
        if progress_path is not None:
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps({
                "status": "running", "processed_structures": processed,
                "total_structures": len(pdb_paths), "last_pdb": path.name,
                "pairs_collected": len(distances),
            }, indent=2) + "\n")
            temporary.replace(progress_path)
    distances_np = np.asarray(distances, dtype=float)
    overlaps_np = np.asarray(overlaps, dtype=float)
    positive = overlaps_np[overlaps_np > 0.0]
    reference_overlap = float(np.median(positive)) if positive.size else 0.1
    return {
        "n_structures": len(pdb_paths),
        "structures_with_contacts": structures_with_contacts,
        "n_nonbonded_pairs_within_cutoff": int(distances_np.size),
        "n_positive_overlap_pairs": int(positive.size),
        "distance_percentiles_A": {str(p): float(np.percentile(distances_np, p)) for p in (1, 5, 25, 50, 75, 95, 99)} if distances_np.size else {},
        "positive_overlap_percentiles_A": {str(p): float(np.percentile(positive, p)) for p in (25, 50, 75, 95)} if positive.size else {},
        "reference_overlap_A": reference_overlap,
        "distance_definition": "primary deposited altloc state, heavy atoms, within 4.5 A, excluding graph-distance 1-2 and 1-3",
        "threshold_definition": "0.75 times the sum of element van der Waals radii",
    }
