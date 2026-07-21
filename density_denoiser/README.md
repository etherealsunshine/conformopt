# Density denoiser

This package builds paired crystallographic patches, trains a residual 3D U-Net,
and evaluates it on proteins that never appear in training or validation.

## Scientific conventions

- The supplied `/train` and `/test` protein split is preserved.
- A validation set is selected by **whole PDB ID** from `/train`; `/test` is not
  used for early stopping.
- Deposited structure-factor CIFs are downloaded from the RCSB and cached before
  conversion to MTZ. Missing/non-X-ray/incomplete reflection sets are logged and
  skipped.
- Both A and B sidechains are omitted when phases are calculated.
- `omit_mfo_dfc` inputs use a sidechain-only synthetic target because a difference
  map does not contain the full modeled backbone/neighborhood density.
- `omit_2mfo_dfc` inputs use a full local synthetic target.
- Deposited occupancies and B factors are retained. Negative, single-conformer
  sidechains are included at a default 4:1 ratio to altloc sites.
- Every acquired protein, generated site, epoch, and evaluation artifact is saved
  immediately. Commands resume by default; use `--overwrite` only intentionally.

## Expected pod layout

```text
~/qfit_unet_data/
  train/**/*.pdb
  test/**/*.pdb

~/workspace/
  density_denoiser/
  ...the rest of this workspace...
```

Generated data are isolated under `~/qfit_unet_data/cache`; source PDBs
are never modified.

The tools default to `~/qfit_unet_data`, which resolves to
`/home/dev/qfit_unet_data` on the current Astera pod. Set `QFIT_UNET_DATA` or
pass `--data-root` explicitly when using another layout.

## Pod setup

From the repository directory inside the pod:

```bash
python -m pip install 'gemmi>=0.6.7' 'numpy>=1.26' 'torch>=2.2' matplotlib
```

## 1. Acquire deposited reflections and create MTZ files

```bash
cd ~/workspace
python -m density_denoiser.data_pipeline acquire \
  --split both \
  --workers 8
```

The RCSB source file is cached as `PDBID-sf.cif`; the derived MTZ is stored next
to it in the cache. Per-protein status JSON makes retries safe.

## 2. Generate paired patches

Start with omit mFo-DFc, the strongest raw map in the five-site experiment:

```bash
python -m density_denoiser.data_pipeline prepare \
  --split both \
  --map-type omit_mfo_dfc \
  --workers 8 \
  --negatives-per-altloc 4
```

Each pair is `(1, 32, 32, 32)` at 0.5 A spacing and is normalized independently.
The manifest is written only from completed site records.

### Residue-frame canonical patches

The paper-quality variant removes arbitrary crystal-frame orientation before the
network sees a patch. Its origin is C-alpha; the x axis is C-alpha to C-beta,
the z axis is `(C-alpha to N) x (C-alpha to C-beta)`, and `y = z x x`. Shared
backbone atoms are preferred; alternate atoms are occupancy-averaged when a
shared atom is unavailable. Every saved transform is checked for orthonormality
and right-handedness.

Canonical outputs are isolated under `cache/{split}/canonical`, so generating
them never overwrites the crystal-frame baseline:

```bash
python -m density_denoiser.data_pipeline prepare \
  --split both \
  --map-type omit_mfo_dfc \
  --frame residue \
  --workers 20 \
  --negatives-per-altloc 4
```

The input map is sampled on the residue-aligned grid and the synthetic target
and local mask are evaluated directly on that same grid. Metadata stores the
crystal-space patch center and the 3x3 crystal-to-local transform. Full cube
rotation augmentation is disabled by default for residue-frame training, while
translation and noise augmentation remain enabled.

## 3. Train

The corrected U-Net has a final 16^3-to-32^3 decoder stage that was missing from
the experiment prompt. A conservative initial run is:

```bash
python -m density_denoiser.train \
  --base-channels 16 \
  --batch-size 8 \
  --epochs 100 \
  --workers 4 \
  --resume
```

For canonical data, add `--frame residue`. Its checkpoints default to
`density_denoiser/model_canonical`, separate from the baseline model.

Increase `--base-channels` to 32 only after checking GPU memory. `denoiser_last.pt`
is replaced atomically after every epoch and `denoiser_best.pt` tracks held-out
protein validation L2.

## 4. Evaluate on untouched test proteins

```bash
python -m density_denoiser.evaluate
```

Outputs include per-site L2, global Pearson correlation, atom-local correlation,
identity and Gaussian-blur baselines, and experimental/denoised/synthetic figures.

## Gates before optimizer integration

Do not run the expensive 50-start optimizer sweep until the denoiser beats both
identity and Gaussian blur on held-out altloc sites. The optimizer and tmol audit
remain downstream validation; reconstruction quality alone is not evidence of
correct conformer recovery.
