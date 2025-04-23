# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "einops",
#   "numpy",
#   "torch",
#   "torch-grid-utils>=0.0.4",
#   "torch-fourier-slice",
#   "torch-fourier-filter",
#   "scipy",
#   "napari[pyqt5]",
#   "torch-refine-tilt-axis-angle",
# ]
# exclude-newer = "2025-04-T00:00:00Z"
# [tool.uv.sources]
# torch-refine-tilt-axis-angle = { path = "../" }
# ///
import einops
import napari
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_fourier_filter.bandpass import low_pass_filter
from torch_fourier_slice import backproject_2d_to_3d, project_3d_to_2d
from torch_grid_utils import circle, coordinate_grid

from torch_refine_tilt_axis_angle import refine_tilt_axis_angle


def simulate_volume() -> torch.Tensor:  # (128, 128, 128)
    point_positions = (
        [64, 32, 32],
        [64, 32, 96],
        [64, 96, 32],
        [64, 96, 96],
    )

    volume = torch.zeros(size=(128, 128, 128))
    for point in point_positions:
        _volume = coordinate_grid((128, 128, 128), center=point, norm=True)
        _volume = _volume < 2
        _volume = _volume.float()
        volume += _volume

    return volume


def simulate_tilt_series(
    volume: torch.Tensor,
    tilt_axis_angle: float,
    tilt_angles: torch.Tensor,
) -> torch.Tensor:
    # construct rotation matrices
    in_plane_rotation_angles = einops.repeat(
        np.array([tilt_axis_angle]), "1 -> b", b=len(tilt_angles)
    )
    euler_angles = einops.rearrange(
        [tilt_angles, in_plane_rotation_angles], pattern="angles b -> b angles"
    )
    rotation_matrices = (
        R.from_euler(seq="yz", angles=euler_angles, degrees=True).inv().as_matrix()
    )
    rotation_matrices = torch.tensor(rotation_matrices).float()

    # make tilt series
    tilt_series = project_3d_to_2d(volume, rotation_matrices=rotation_matrices)
    return tilt_series


def reconstruct_tomogram(
    tilt_series: torch.Tensor,
    tilt_axis_angle: float,
    tilt_angles: torch.Tensor,
) -> torch.Tensor:
    # construct rotation matrices
    in_plane_rotation_angles = einops.repeat(
        np.array([tilt_axis_angle]), "1 -> b", b=len(tilt_angles)
    )
    euler_angles = einops.rearrange(
        [tilt_angles, in_plane_rotation_angles], pattern="angles b -> b angles"
    )
    rotation_matrices = (
        R.from_euler(seq="yz", angles=euler_angles, degrees=True).inv().as_matrix()
    )
    rotation_matrices = torch.tensor(rotation_matrices).float()

    # make tilt series
    reconstruction = backproject_2d_to_3d(
        tilt_series, rotation_matrices=rotation_matrices
    )
    return reconstruction


if __name__ == "__main__":
    # simulate volume with a few points in the xy plane
    volume = simulate_volume()
    lpf = low_pass_filter(
        0.2,
        0.05,
        volume.shape,
        rfft=True,
        fftshift=False,
    )
    volume = torch.fft.irfftn(torch.fft.rfftn(volume) * lpf)

    # setup tilt series geometry
    tilt_axis_angle = 85
    tilt_axis_angle_initial = 70
    tilt_angles = np.linspace(-60, 60, num=61, endpoint=True)
    alignment_mask = circle(56, (128, 128), smoothing_radius=8)

    # simulate tilt series
    tilt_series = simulate_tilt_series(
        volume,
        tilt_axis_angle,
        tilt_angles,
    )
    # make reconstruction with initial (bad) guess
    recon_before = reconstruct_tomogram(
        tilt_series, tilt_axis_angle_initial, tilt_angles
    )

    # run torch-tiltxcorr and apply shifts
    optimized_tilt_axis_angle = refine_tilt_axis_angle(
        tilt_series=tilt_series,
        alignment_mask=torch.ones(tuple(tilt_series.shape[-2:])),
        tilt_axis_angle=tilt_axis_angle_initial,
    )

    # reconstruct after optimization
    recon_after = reconstruct_tomogram(
        tilt_series, optimized_tilt_axis_angle, tilt_angles
    )

    # visiualize results
    viewer = napari.Viewer()
    viewer.add_image(recon_before)
    viewer.add_image(recon_after)
    napari.run()
