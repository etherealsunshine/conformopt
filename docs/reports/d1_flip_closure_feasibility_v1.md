# D1 flip-panel closure feasibility

**Status:** complete. Measurements use deposited A/B coordinates only; no qFit sampling, density fitting, or mirror descent was run.

## 7UTC A:ARG52

### 1. Anchor agreement — decisive check

Maximum A/B backbone-coordinate difference at each window position:

| Position | Max Δ{N,CA,C,O} (Å) |
|---:|---:|
| −3 | 0.000 |
| −2 | 0.000 |
| −1 | 0.467 |
| 0 | 2.304 |
| +1 | 1.141 |
| +2 | 1.493 |
| +3 | **1.573** |

The +3 anchor is not fixed.  Thus the deposited A→B window is not a
closure-preserving move under fixed end anchors: the near-zero projected A→B
tangent is a **finding**, not a Jacobian bug.

### 2. Forward kinematics

Applying the wrapped deposited `delta_q` to A with qFit's `BackboneRotator`
gives RMSD to B of **0.427 Å** for central {N,CA,C,O} and **1.159 Å** across
the full seven-residue backbone window.  It is not the deposited transition.

### 3. Dihedral wrapping

The raw 14-component φ/ψ delta contains `−261.869°`; its wrapped value is
`+98.131°`.  All components were wrapped to (−180°, 180°].

### 4. Omega (qFit does not vary omega)

| Peptide | \|Δω\| (°) |
|---|---:|
| −3→−2 | 0.000 |
| −2→−1 | 6.412 |
| −1→0 | 13.117 |
| 0→+1 | 11.407 |
| +1→+2 | 3.098 |
| +2→+3 | **14.026** |

### 5. Covalent geometry

- Max backbone bond-length difference: **0.0090 Å** (`C(0)–N(+1)`).
- Max backbone angle difference: **3.386°** (`N–CA–C` at +1).
- Conservative bond-only full-window RMSD floor: **0.0012 Å**.
- Angle-and-bond-aware local-triangle full-window floor: **0.0091 Å**.

Therefore covalent geometry is not the source of the 1.159 Å full-window
forward-kinematics miss; the moving remote anchors and omega changes are.

## All 33 flip sites: centre±3 anchor distribution

Only **19/33** have a complete sequential A/B seven-residue window. Fourteen
are not closure-testable under this parameterisation (truncated qFit segment
or B window).  Among the 19 testable sites, the maximum of the ±3 anchor
disagreement is:

| Statistic | Å |
|---|---:|
| Minimum | 0.000 |
| Median | **0.805** |
| Maximum | **1.573** |
| >0.1 Å | **17 / 19** |

Only **7ZTL A:ILE257** and **1RWR A:ASN294** have exactly matching ±3
backbone anchors.  Thus the great majority of the closure-testable flip panel
is not reachable-in-principle as a fixed-anchor, φ/ψ-only, seven-residue
closure move.

Pod artifacts:

- `/home/dev/qfit_unet_data/qfit_audit/d1_flip_closure_7utc_full_v2`
- `/home/dev/qfit_unet_data/qfit_audit/d1_flip_closure_anchors33_v2`
