"""Small PDB/connectivity helpers shared by the A-prime clash calculation.

The optimizer uses these helpers to remove bonded 1-2 and 1-3 contacts from
its non-bonded contact graph.  They deliberately accept an ordinary PDB and a
CCP4 monomer-library directory so the same filtering rule can be used with
different Phenix installations.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import gemmi


def read_pdb_atoms(path: Path) -> list[dict]:
    """Read the atom records needed for contact topology and coordinates."""
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append({
            "chain": line[21].strip(),
            "resseq": line[22:26].strip(),
            "icode": line[26].strip(),
            "resname": line[17:20].strip().upper(),
            "name": line[12:16].strip().upper(),
            "altloc": line[16].strip(),
            "element": (line[76:78].strip() or line[12:16].strip()[:1]).upper(),
            "xyz": tuple(float(line[i:i + 8]) for i in (30, 38, 46)),
        })
    return atoms


class Connectivity:
    """Build residue and peptide connectivity from CCP4 monomer definitions."""

    def __init__(self, pdb: Path, monomer_root: Path):
        self.adj: dict[tuple, set[tuple]] = defaultdict(set)
        self.monomer_root = Path(monomer_root)
        residues: dict[tuple, dict] = {}
        order: dict[str, list[tuple]] = defaultdict(list)
        for atom in read_pdb_atoms(pdb):
            residue_key = (
                atom["chain"], atom["resseq"], atom["icode"], atom["resname"]
            )
            residues.setdefault(residue_key, {})[atom["name"]] = atom
            if residue_key not in order[atom["chain"]]:
                order[atom["chain"]].append(residue_key)

        cache: dict[str, list[tuple[str, str]]] = {}
        for residue_key in residues:
            resname = residue_key[3]
            cache.setdefault(resname, self._monomer_bonds(resname))
            for atom_1, atom_2 in cache[resname]:
                key_1 = residue_key[:3] + (atom_1,)
                key_2 = residue_key[:3] + (atom_2,)
                self.adj[key_1].add(key_2)
                self.adj[key_2].add(key_1)

        # Peptide C-N bonds are not represented as intra-monomer CIF bonds.
        for chain_residues in order.values():
            for first, second in zip(chain_residues, chain_residues[1:]):
                try:
                    sequential = int(second[1]) - int(first[1]) in (0, 1)
                except ValueError:
                    sequential = False
                if not sequential or first[3] in ("HOH", "WAT") or second[3] in ("HOH", "WAT"):
                    continue
                key_1 = first[:3] + ("C",)
                key_2 = second[:3] + ("N",)
                self.adj[key_1].add(key_2)
                self.adj[key_2].add(key_1)

    def _monomer_bonds(self, resname: str) -> list[tuple[str, str]]:
        cif_path = self.monomer_root / resname[0].lower() / f"data_{resname}.cif"
        if not cif_path.exists():
            return []
        try:
            block = gemmi.cif.read(str(cif_path))[resname]
            loop = block.find_loop("_chem_comp_bond.atom_id_1").get_loop()
            tags, values = list(loop.tags), list(loop.values)
            width = len(tags)
            first = tags.index("_chem_comp_bond.atom_id_1")
            second = tags.index("_chem_comp_bond.atom_id_2")
            return [
                (values[i + first].strip().upper(), values[i + second].strip().upper())
                for i in range(0, len(values), width)
            ]
        except Exception:
            return []

    def distance(self, atom_1: dict, atom_2: dict, max_depth: int = 2) -> int | None:
        """Return graph distance up to ``max_depth``; otherwise ``None``."""
        key_1 = (atom_1["chain"], atom_1["resseq"], atom_1["icode"], atom_1["name"])
        key_2 = (atom_2["chain"], atom_2["resseq"], atom_2["icode"], atom_2["name"])
        if key_1 == key_2:
            return 0
        queue = deque([(key_1, 0)])
        seen = {key_1}
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self.adj.get(node, ()):
                if neighbor == key_2:
                    return depth + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return None
