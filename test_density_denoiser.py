from pathlib import Path
from dataclasses import asdict

import gemmi
import torch

import numpy as np

from density_denoiser.data_pipeline import (
    _calculate_fcalc,
    _grid_coordinates,
    _omit_sidechain,
    _patch_transform,
    _sidechain_model,
    discover_sites,
    normalize_patch,
    residue_frame,
    synthetic_patch,
)
from density_denoiser.model import ResidualDensityDenoiser, spatial_gradient
from density_denoiser.five_site_optimizer import _canonical_centers
from density_denoiser.landscape import (
    landscape_distillation_loss,
    radial_mask,
    render_candidates,
)
from density_denoiser.prepare_landscape_cache import _build_site


def test_unet_preserves_32_cube_shape():
    model = ResidualDensityDenoiser(base_channels=2)
    inputs = torch.randn(1, 1, 32, 32, 32)
    assert model(inputs).shape == inputs.shape


def test_spatial_gradient_shape():
    inputs = torch.randn(2, 1, 32, 32, 32)
    assert spatial_gradient(inputs).shape == (2, 3, 32, 32, 32)


def test_physics_rotamer_centers_include_residue_specific_planarity():
    assert np.allclose(np.rad2deg(_canonical_centers("ASP", 1)), [0.0, 180.0, -180.0])
    assert np.allclose(np.rad2deg(_canonical_centers("ARG", 3)), [-90.0, 90.0])
    assert np.allclose(
        np.rad2deg(_canonical_centers("MET", 2)), [-90.0, 90.0, 180.0, -180.0]
    )


def test_2o1k_sidechain_altloc_discovery():
    structure = gemmi.read_structure(str(Path(__file__).parent / "data" / "2O1K.pdb"))
    sites = discover_sites(structure, "2O1K", "test", negatives_per_altloc=0, seed=1)
    assert {site.key for site in sites} == {
        "2O1K_A_MET112",
        "2O1K_A_ARG129",
        "2O1K_B_MET112",
        "2O1K_B_ASP114",
        "2O1K_B_ARG129",
    }


def test_sidechain_subtraction_exactly_matches_full_omit_calculation():
    root = Path(__file__).parent
    structure = gemmi.read_structure(str(root / "data" / "2O1K.pdb"))
    mtz = gemmi.read_mtz_file(str(root / "data" / "2O1K.mtz"))
    site = next(
        site for site in discover_sites(structure, "2O1K", "test", 0, 1)
        if site.key == "2O1K_A_MET112"
    )
    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    miller = mtz.make_miller_array()
    full = _calculate_fcalc(calculator, structure[0], miller)
    sidechain = _calculate_fcalc(calculator, _sidechain_model(structure, site), miller)
    direct_omit = _calculate_fcalc(calculator, _omit_sidechain(structure, site)[0], miller)
    np.testing.assert_allclose(full - sidechain, direct_omit, rtol=1e-5, atol=1e-3)


def test_residue_frame_is_right_handed_and_patch_transform_matches_coordinates():
    root = Path(__file__).parent
    structure = gemmi.read_structure(str(root / "data" / "2O1K.pdb"))
    site = next(
        site for site in discover_sites(structure, "2O1K", "test", 0, 1)
        if site.key == "2O1K_A_MET112"
    )
    origin, rotation = residue_frame(structure, site)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)
    coordinates = _grid_coordinates(origin, 7, 0.5, rotation)
    transform = _patch_transform(origin, 7, 0.5, rotation)
    for index in ((0, 0, 0), (3, 3, 3), (6, 2, 5)):
        transformed = transform.apply(gemmi.Vec3(*map(float, index)))
        np.testing.assert_allclose(
            [transformed.x, transformed.y, transformed.z], coordinates[index], atol=1e-6
        )


def test_canonical_synthetic_patch_is_invariant_to_arbitrary_rigid_rotation():
    root = Path(__file__).parent
    structure = gemmi.read_structure(str(root / "data" / "2O1K.pdb"))
    site = next(
        site for site in discover_sites(structure, "2O1K", "test", 0, 1)
        if site.key == "2O1K_A_MET112"
    )
    origin, frame = residue_frame(structure, site)
    reference = synthetic_patch(
        structure, site, 32, 0.5, "sidechain", origin, frame
    )

    axis = np.asarray([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(37.0)
    cross = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    rigid_rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    translation = np.asarray([7.0, -11.0, 4.5])
    rotated = structure.clone()
    for chain in rotated[0]:
        for residue in chain:
            for atom in residue:
                position = rigid_rotation @ np.asarray(atom.pos.tolist()) + translation
                atom.pos = gemmi.Position(*position)
    rotated_origin, rotated_frame = residue_frame(rotated, site)
    transformed = synthetic_patch(
        rotated, site, 32, 0.5, "sidechain", rotated_origin, rotated_frame
    )
    np.testing.assert_allclose(reference, transformed, rtol=2e-5, atol=2e-5)


def test_landscape_native_candidate_reproduces_synthetic_target():
    pdb = Path(__file__).parent / "data" / "2O1K.pdb"
    structure = gemmi.read_structure(str(pdb))
    site = next(
        site for site in discover_sites(structure, "2O1K", "test", 0, 41)
        if site.key == "2O1K_A_MET112"
    )
    label = _build_site(
        {**asdict(site), "key": site.key, "pair_path": "unused.npz"},
        structure,
        seed=20260720,
    )
    assert label is not None
    candidates = render_candidates(
        torch.from_numpy(label["positions"])[None],
        torch.from_numpy(label["sigma2"])[None],
        torch.from_numpy(label["weights"])[None],
        torch.from_numpy(label["atom_mask"])[None],
    )
    expected = normalize_patch(synthetic_patch(
        structure, site, 32, 0.5, "sidechain"
    ))[0]
    assert np.mean((candidates[0, 0].numpy() - expected) ** 2) < 1e-8


def test_landscape_loss_is_zero_for_oracle_density():
    positions = torch.zeros((1, 2, 1, 3))
    positions[0, 1, 0, 0] = 2.0
    sigma2 = torch.full((1, 2, 1), 0.2)
    weights = torch.ones((1, 2, 1))
    atom_mask = torch.ones((1, 2, 1), dtype=torch.bool)
    candidates = render_candidates(
        positions, sigma2, weights, atom_mask, size=8, spacing=0.5
    )
    target = candidates[:, :1]
    mask = radial_mask(8, 0.5, 2.0, device=torch.device("cpu"))
    loss, metrics = landscape_distillation_loss(target, target, candidates, mask)
    assert float(loss) < 1e-8
    assert float(metrics["native_top1"]) == 1.0
    assert float(metrics["oracle_native_top1"]) == 1.0
