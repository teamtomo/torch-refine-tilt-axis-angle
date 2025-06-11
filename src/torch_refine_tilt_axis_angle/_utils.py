import torch
import einops
import scipy.ndimage as ndi
from typing import Sequence
from torch_grid_utils import coordinate_grid


def _tilt_axis_grid(
        image_shape: tuple[int, int],
        tilt_axis_angles: torch.Tensor,
        device: torch.device | None = None,
) -> torch.FloatTensor:
    """Calculate absolute perpendicular distance from a line at a given angle.

    Parameters
    ----------
    image_shape: Sequence[int]
        Shape of the 2D image (height, width).
    tilt_axis_angles: float
        Angle of the line in degrees. 0 means horizontal line.
        Positive angles rotate counterclockwise.
    device: torch.device
        PyTorch device on which to put the result.

    Returns
    -------
    distances: torch.FloatTensor
        Array of shape `image_shape` containing absolute perpendicular distances
        from each pixel to the line.
    """
    line_center = [(image_shape[0] - 1) / 2, (image_shape[1] - 1) / 2]

    # Get coordinate grid centered at the line center
    grid = coordinate_grid(
        image_shape=image_shape,
        center=line_center,
        norm=False,
        device=device
    )  # Shape: (height, width, 2) with coordinates [y, x]

    # Extract y and x coordinates
    y_coords = grid[..., 0]  # (height, width)
    x_coords = grid[..., 1]  # (height, width)

    # Convert angle from degrees to radians
    angles_rad = torch.deg2rad(tilt_axis_angles)
    angles_rad = einops.rearrange(angles_rad, 'n -> n 1 1')

    # Calculate perpendicular distance from line
    # For a line at angle θ, the perpendicular distance is:
    # distance = -x * sin(θ) + y * cos(θ)
    cos_angle = torch.cos(angles_rad)
    sin_angle = torch.sin(angles_rad)

    distances = -x_coords * sin_angle + y_coords * cos_angle
    distances = distances * (2 / image_shape[0])  # norm distance to -1, 1

    # Return absolute distances
    return torch.abs(distances)


def _tilt_sample_mask(
        image_shape: tuple[int, int],
        tilt_axis_angles: torch.Tensor,
        tilt_angles: torch.Tensor,
        device: torch.device | None = None,
) -> torch.Tensor:
    grids = _tilt_axis_grid(
        image_shape=image_shape, tilt_axis_angles=tilt_axis_angles, device=device
    )
    factor = torch.cos(torch.deg2rad(tilt_angles))
    factor = einops.rearrange(factor, 'a -> a 1 1')
    mask = (grids <= factor)
    return mask


def _common_line_taper(
        line_shape: int,
        radius: float,
        smoothing_radius: float,
) -> torch.Tensor:
    grid = torch.abs(torch.arange(line_shape) - line_shape / 2)
    mask = torch.zeros_like(grid, dtype=torch.bool)
    mask[grid < radius] = 1
    mask = _add_soft_edge_single_binary_image(mask, smoothing_radius)
    return mask


def _add_soft_edge_single_binary_image(
    image: torch.Tensor, smoothing_radius: float
) -> torch.FloatTensor:
    # move explicitly to cpu for scipy
    distances = ndi.distance_transform_edt(torch.logical_not(image).to("cpu"))
    distances = torch.as_tensor(distances, device=image.device).float()
    idx = torch.logical_and(distances > 0, distances <= smoothing_radius)
    output = torch.clone(image).float()
    output[idx] = torch.cos((torch.pi / 2) * (distances[idx] / smoothing_radius))
    return output