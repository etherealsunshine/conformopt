from __future__ import annotations

import math

import torch


AUDIT_RULE_VERSION = (
    "2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2"
)


# Production sidechain topology for every non-ring standard residue with at
# least one chi angle. PRO is excluded because independent bond rotations do
# not preserve its pyrrolidine ring closure; ALA and GLY have no sidechain chi.
CHI_SPECS = {
    "ARG": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                      ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
        "rotations": (("CA", "CB", ("CG", "CD", "NE", "CZ", "NH1", "NH2")),
                      ("CB", "CG", ("CD", "NE", "CZ", "NH1", "NH2")),
                      ("CG", "CD", ("NE", "CZ", "NH1", "NH2")),
                      ("CD", "NE", ("CZ", "NH1", "NH2"))),
    },
    "ASN": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        "rotations": (("CA", "CB", ("CG", "OD1", "ND2")),
                      ("CB", "CG", ("OD1", "ND2"))),
    },
    "ASP": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        "rotations": (("CA", "CB", ("CG", "OD1", "OD2")),
                      ("CB", "CG", ("OD1", "OD2"))),
    },
    "CYS": {
        "dihedrals": (("N", "CA", "CB", "SG"),),
        "rotations": (("CA", "CB", ("SG",)),),
    },
    "GLN": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                      ("CB", "CG", "CD", "OE1")),
        "rotations": (("CA", "CB", ("CG", "CD", "OE1", "NE2")),
                      ("CB", "CG", ("CD", "OE1", "NE2")),
                      ("CG", "CD", ("OE1", "NE2"))),
    },
    "GLU": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                      ("CB", "CG", "CD", "OE1")),
        "rotations": (("CA", "CB", ("CG", "CD", "OE1", "OE2")),
                      ("CB", "CG", ("CD", "OE1", "OE2")),
                      ("CG", "CD", ("OE1", "OE2"))),
    },
    "HIS": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
        "rotations": (("CA", "CB", ("CG", "ND1", "CD2", "CE1", "NE2")),
                      ("CB", "CG", ("ND1", "CD2", "CE1", "NE2"))),
    },
    "ILE": {
        "dihedrals": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
        "rotations": (("CA", "CB", ("CG1", "CG2", "CD1")),
                      ("CB", "CG1", ("CD1",))),
    },
    "LEU": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "rotations": (("CA", "CB", ("CG", "CD1", "CD2")),
                      ("CB", "CG", ("CD1", "CD2"))),
    },
    "LYS": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                      ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
        "rotations": (("CA", "CB", ("CG", "CD", "CE", "NZ")),
                      ("CB", "CG", ("CD", "CE", "NZ")),
                      ("CG", "CD", ("CE", "NZ")),
                      ("CD", "CE", ("NZ",))),
    },
    "MET": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
                      ("CB", "CG", "SD", "CE")),
        "rotations": (("CA", "CB", ("CG", "SD", "CE")),
                      ("CB", "CG", ("SD", "CE")),
                      ("CG", "SD", ("CE",))),
    },
    "PHE": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "rotations": (("CA", "CB", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")),
                      ("CB", "CG", ("CD1", "CD2", "CE1", "CE2", "CZ"))),
    },
    "SER": {
        "dihedrals": (("N", "CA", "CB", "OG"),),
        "rotations": (("CA", "CB", ("OG",)),),
    },
    "THR": {
        "dihedrals": (("N", "CA", "CB", "OG1"),),
        "rotations": (("CA", "CB", ("OG1", "CG2")),),
    },
    "TRP": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "rotations": (("CA", "CB", ("CG", "CD1", "CD2", "NE1", "CE2", "CE3",
                                      "CZ2", "CZ3", "CH2")),
                      ("CB", "CG", ("CD1", "CD2", "NE1", "CE2", "CE3", "CZ2",
                                      "CZ3", "CH2"))),
    },
    "TYR": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "rotations": (("CA", "CB", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")),
                      ("CB", "CG", ("CD1", "CD2", "CE1", "CE2", "CZ", "OH"))),
    },
    "VAL": {
        "dihedrals": (("N", "CA", "CB", "CG1"),),
        "rotations": (("CA", "CB", ("CG1", "CG2")),),
    },
}


# Alternative names describe chemically equivalent terminal atoms. RMSD is
# minimized over all applicable swaps so atom naming cannot create false misses.
EQUIVALENT_ATOM_SWAP_GROUPS = {
    "ARG": ((("NH1", "NH2"),),),
    "ASP": ((("OD1", "OD2"),),),
    "GLU": ((("OE1", "OE2"),),),
    "LEU": ((("CD1", "CD2"),),),
    "PHE": ((("CD1", "CD2"), ("CE1", "CE2")),),
    "TYR": ((("CD1", "CD2"), ("CE1", "CE2")),),
    "VAL": ((("CG1", "CG2"),),),
}


def canonical_centers_radians(resname: str, chi_index: int) -> tuple[float, ...]:
    """Chemistry-aware centers shared by the soft prior and endpoint audit.

    These are deliberately broad marginal centers rather than a joint,
    backbone-conditioned rotamer probability model.
    """
    if (resname, chi_index) in {
        ("ASP", 1), ("ASN", 1), ("GLU", 2), ("GLN", 2),
    }:
        return (0.0, math.pi, -math.pi)
    if (resname, chi_index) == ("ARG", 3):
        return (-math.pi / 2, math.pi / 2, math.pi, -math.pi)
    if (resname, chi_index) == ("TRP", 1):
        return (0.0, math.radians(-105.0), math.radians(105.0))
    if (resname, chi_index) == ("HIS", 1):
        return (
            math.radians(-170.0), math.radians(-80.0),
            math.radians(80.0), math.radians(170.0),
        )
    if resname in {"PHE", "TYR"} and chi_index == 1:
        return (-math.pi / 2, math.pi / 2)
    if resname == "MET" and chi_index == 2:
        return (-math.pi / 2, math.pi / 2, math.pi, -math.pi)
    return (-math.pi / 3, math.pi / 3, math.pi, -math.pi)


def canonical_width_degrees(resname: str, chi_index: int) -> float:
    """Allowed marginal deviation for the shared rotamer guardrail.

    Chi1 remains tight, internal chis are moderately broad, and chemically
    broad terminal amide/carboxyl torsions are effectively exempt because the
    nearest 0/180-degree center is always at most 90 degrees away.
    """
    if (resname, chi_index) in {
        ("ASP", 1), ("ASN", 1), ("GLU", 2), ("GLN", 2),
    }:
        return 90.0
    if resname in {"PHE", "TYR"} and chi_index == 1:
        return 45.0
    if chi_index == 0:
        return 45.0
    if chi_index == len(CHI_SPECS[resname]["dihedrals"]) - 1:
        return 60.0
    return 45.0


def angular_delta_degrees(value: float, target: float) -> float:
    return abs(((value - target + 180.0) % 360.0) - 180.0)


def classify_rotamer_degrees(
    resname: str, angles: list[float]
) -> tuple[str, list[float], list[float], bool]:
    """Classify marginal chis using the shared centers and per-chi widths."""
    labels, deviations, widths = [], [], []
    for index, angle in enumerate(angles):
        label, center = min(
            canonical_centers_degrees(resname, index),
            key=lambda item: angular_delta_degrees(angle, item[1]),
        )
        labels.append(label)
        deviations.append(angular_delta_degrees(angle, center))
        widths.append(canonical_width_degrees(resname, index))
    return (
        "/".join(labels),
        deviations,
        widths,
        all(deviation <= width for deviation, width in zip(deviations, widths)),
    )


def canonical_centers_degrees(
    resname: str, chi_index: int
) -> list[tuple[str, float]]:
    centers = canonical_centers_radians(resname, chi_index)
    labels = {
        -180.0: "t", -90.0: "m-", -60.0: "g-", 0.0: "p0",
        60.0: "g+", 90.0: "m+", 180.0: "t",
        -105.0: "m-105", 105.0: "m+105",
        -80.0: "m-80", 80.0: "m+80",
        -170.0: "t-170", 170.0: "t+170",
    }
    return [(labels[round(math.degrees(value), 6)], math.degrees(value)) for value in centers]


def reference_permutations(names: list[str], resname: str) -> list[list[int]]:
    """Return index permutations induced by valid equivalent-atom relabelings."""
    identity = list(range(len(names)))
    permutations = [identity]
    for swap_group in EQUIVALENT_ATOM_SWAP_GROUPS.get(resname, ()):
        if any(left not in names or right not in names for left, right in swap_group):
            continue
        expanded = []
        for permutation in permutations:
            expanded.append(permutation)
            swapped = permutation.copy()
            for left, right in swap_group:
                i, j = names.index(left), names.index(right)
                swapped[i], swapped[j] = swapped[j], swapped[i]
            expanded.append(swapped)
        permutations = expanded
    return permutations


def symmetry_aware_rmsd(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    names: list[str],
    resname: str,
) -> torch.Tensor:
    """Conventional heavy-atom RMSD minimized over equivalent atom labels."""
    values = [
        torch.sqrt(torch.mean(torch.sum(
            (candidate - reference[permutation]).square(), dim=-1
        )))
        for permutation in reference_permutations(names, resname)
    ]
    return torch.stack(values).min()
