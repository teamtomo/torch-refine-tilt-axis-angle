import einops
import torch

# teamtomo torch functionality
from torch_cubic_spline_grids import CubicBSplineGrid1d
from torch_fourier_slice import project_2d_to_1d
from torch_affine_utils.transforms_2d import R


def refine_tilt_axis_angle(
        tilt_series: torch.Tensor,
        alignment_mask: torch.Tensor,
        initial_tilt_axis_angle: torch.Tensor | float,
        grid_points: int = 1,
        iterations: int = 3,
) -> torch.Tensor:
    """Optimize tilt axis angles using a cubic B-spline grid and LBFGS optimization.

    This function optimizes the tilt axis angles for electron tomography by minimizing
    the differences between common line projections across the tilt series.

    Parameters
    ----------
    tilt_series : torch.Tensor
        Tensor containing the tilt series images with shape [n_tilts, height, width].
    alignment_mask : torch.Tensor
        Binary mask of the same shape as tilt_series, indicating regions to consider
        for alignment.
    initial_tilt_axis_angle : torch.Tensor or float
        Initial guess for the tilt axis angle(s) in degrees.
    grid_points : int, default=1
        Number of control points for the cubic B-spline grid.
    iterations : int, default=3
        Number of optimization iterations to perform.

    Returns
    -------
    torch.Tensor
        Optimized tilt axis angles for each tilt in the series, in degrees.

    Notes
    -----
    The optimization works by:
    1. Projecting images perpendicular to the tilt axis
    2. Comparing these projections across different tilts
    3. Minimizing differences between projections using LBFGS
    """
    if isinstance(initial_tilt_axis_angle, float):
        initial_tilt_axis_angle = [initial_tilt_axis_angle,] * grid_points
    else:
        n_angles = len(initial_tilt_axis_angle)
        if n_angles == 1:
            initial_tilt_axis_angle = [initial_tilt_axis_angle, ] * grid_points
        elif n_angles != grid_points:
            raise ValueError(
                "Initial tilt axis angles must have length 1 or the same "
                "length as the number of grid points."
            )

    n_tilts = tilt_series.shape[0]
    device = tilt_series.device
    tilt_series = tilt_series * alignment_mask

    # generate a weighting for the common line ROI by projecting the mask
    mask_weights = project_2d_to_1d(
        tilt_series,
        torch.eye(2, device=device),  # angle does not matter
    )
    mask_weights = einops.rearrange(mask_weights, '1 w -> w')
    mask_weights = mask_weights / mask_weights.max()  # normalise to 0 and 1

    # optimize tilt axis angle
    tilt_axis_grid = CubicBSplineGrid1d(resolution=grid_points, n_channels=1)
    tilt_axis_grid.data = torch.tensor(
        [torch.mean(initial_tilt_axis_angle),] * grid_points,
        dtype=torch.float32,
        device=device,
    )
    interpolation_points = torch.linspace(
        0, 1, n_tilts, device=device
    )

    lbfgs = torch.optim.LBFGS(
        tilt_axis_grid.parameters(),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        # The common line is the projection perpendicular to the
        # tilt-axis, hence add 90 degrees to project along the x-axis
        pred_tilt_axis_angles = tilt_axis_grid(interpolation_points) + 90
        M = R(pred_tilt_axis_angles, yx=False)
        M = M[:, :2, :2]  # we only need the rotation matrix

        projections = torch.cat(
            [
                project_2d_to_1d(tilt_series[[i]], M[[i]])
                for i in range(n_tilts)
            ]
        )
        projections = projections - einops.reduce(
            projections, "tilt w -> tilt 1", reduction="mean"
        )
        projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        projections = projections * mask_weights  # weight the common lines

        lbfgs.zero_grad()
        squared_differences = (
            projections - einops.rearrange(projections, "b d -> b 1 d")
        ) ** 2
        loss = einops.reduce(squared_differences, "b1 b2 d -> 1", reduction="sum")
        loss.backward()
        return loss

    for _ in range(iterations):
        lbfgs.step(closure)

    tilt_axis_angles = tilt_axis_grid(interpolation_points)

    return tilt_axis_angles.detach()