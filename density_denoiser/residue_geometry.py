from __future__ import annotations

import math

import torch


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
    """Simple chemistry-aware centers used by the validated soft prior.

    The prior remains intentionally broad. It is a guardrail against torsions in
    disallowed regions, not a probability model replacing the density target.
    """
    if (resname, chi_index) in {
        ("ASP", 1), ("ASN", 1), ("GLU", 2), ("GLN", 2),
    }:
        return (0.0, math.pi, -math.pi)
    if (resname, chi_index) in {
        ("ARG", 3), ("HIS", 1), ("PHE", 1), ("TRP", 1), ("TYR", 1),
    }:
        return (-math.pi / 2, math.pi / 2)
    if resname == "MET" and chi_index == 2:
        return (-math.pi / 2, math.pi / 2, math.pi, -math.pi)
    return (-math.pi / 3, math.pi / 3, math.pi, -math.pi)


def canonical_centers_degrees(
    resname: str, chi_index: int
) -> list[tuple[str, float]]:
    centers = canonical_centers_radians(resname, chi_index)
    labels = {
        -180.0: "t", -90.0: "m-", -60.0: "g-", 0.0: "p0",
        60.0: "g+", 90.0: "m+", 180.0: "t",
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
