import json

import numpy as np

from gaussian_decomp.data import GaussianCloud, load_gaussian_cloud, save_npz
from gaussian_decomp.fitting import fit_baseline, fit_gaussian_aware
from gaussian_decomp.primitives import SuperFrustum, Superquadric, load_primitives, save_primitives
from gaussian_decomp.synthetic import make_synthetic_cloud
from gaussian_decomp.evaluate import part_consistency, scene_metrics


def test_cloud_covariance_and_npz_roundtrip(tmp_path):
    cloud, _ = make_synthetic_cloud(12)
    assert cloud.covariance.shape == (36, 3, 3)
    assert np.all(np.linalg.eigvalsh(cloud.covariance) > 0)
    path = tmp_path / "cloud.npz"
    save_npz(cloud, path)
    loaded = load_gaussian_cloud(path)
    np.testing.assert_allclose(cloud.positions, loaded.positions)


def test_ascii_3dgs_ply_loader(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n0 0 0 0.8 -1 -1 -1 1 0 0 0\n1 0 0 0.9 -1 -1 -1 1 0 0 0\n"
    )
    loaded = load_gaussian_cloud(path)
    assert loaded.size == 2
    np.testing.assert_allclose(loaded.positions[:, 0], [0, 1])
    np.testing.assert_allclose(loaded.weights, [0.8, 0.9])


def test_shapesplat_directory_loader_with_color(tmp_path):
    np.save(tmp_path / "coord.npy", np.zeros((3, 3)))
    np.save(tmp_path / "scale.npy", np.ones((3, 3)) * 0.1)
    np.save(tmp_path / "quat.npy", np.tile([1, 0, 0, 0], (3, 1)))
    np.save(tmp_path / "opacity.npy", np.ones(3) * 0.5)
    np.save(tmp_path / "color.npy", np.ones((3, 3)))
    from gaussian_decomp.data import load_shapesplat_directory
    cloud = load_shapesplat_directory(tmp_path)
    assert cloud.features.shape == (3, 3)


def test_part_consistency_and_scene_metrics():
    cloud, _ = make_synthetic_cloud(20)
    labels = np.repeat([0, 1, 2], 20)
    primitives, _ = fit_gaussian_aware(cloud, n_primitives=3, iterations=4)
    result = part_consistency(cloud.positions, labels, primitives)
    assert 0.0 <= result["purity"] <= 1.0
    assert 0.0 <= result["nmi"] <= 1.0 + 1e-8
    scene = scene_metrics(cloud, primitives)
    assert scene["n_gaussians"] == 60
    assert scene["n_primitives"] == 3


def test_baseline_and_gaussian_aware_fit():
    cloud, _ = make_synthetic_cloud(30)
    baseline, baseline_report = fit_baseline(cloud, n_primitives=3, iterations=5)
    aware, aware_report = fit_gaussian_aware(cloud, n_primitives=3, iterations=5)
    assert len(baseline) == len(aware) == 3
    assert all(isinstance(p, Superquadric) for p in baseline + aware)
    assert baseline_report.method == "point-cloud-baseline"
    assert aware_report.method == "gaussian-aware"
    assert aware_report.attribute_ablation["covariance"]
    assert np.isfinite(aware_report.final_loss)


def test_superfrustum_fit_and_serialization(tmp_path):
    cloud, _ = make_synthetic_cloud(20)
    from gaussian_decomp.fitting import FitConfig, fit_primitives
    primitives, _ = fit_primitives(cloud, FitConfig(n_primitives=3, primitive_type="superfrustum", iterations=3))
    assert all(isinstance(p, SuperFrustum) for p in primitives)
    path = tmp_path / "primitives.json"
    save_primitives(primitives, str(path))
    loaded = load_primitives(str(path))
    assert all(isinstance(p, SuperFrustum) for p in loaded)
