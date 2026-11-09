"""Learning structural primitive decomposition from pretrained 3DGS."""

from .data import GaussianCloud, convert_shapesplat_directory, load_gaussian_cloud, load_shapesplat_directory, save_npz
from .evaluate import assembly_metrics, part_consistency, scene_metrics
from .geometry import load_off, sample_off
from .fitting import FitConfig, FitReport, fit_baseline, fit_gaussian_aware, fit_primitives
from .primitives import SuperFrustum, Superquadric, load_primitives, save_primitives

__all__ = [
    "GaussianCloud", "load_gaussian_cloud", "load_shapesplat_directory", "convert_shapesplat_directory", "save_npz", "FitConfig", "FitReport",
    "fit_baseline", "fit_gaussian_aware", "fit_primitives", "Superquadric",
    "SuperFrustum", "save_primitives", "load_primitives", "assembly_metrics", "part_consistency", "scene_metrics", "load_off", "sample_off",
]
