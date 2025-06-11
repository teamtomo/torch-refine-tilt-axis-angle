"""Refine an initial tilt axis angle."""

import einops
import torch
from torch_affine_utils.transforms_2d import R

# teamtomo torch functionality
from torch_cubic_spline_grids import CubicBSplineGrid1d
from torch_fourier_slice import project_2d_to_1d

from ._utils import _tilt_sample_mask, _common_line_taper


def refine_tilt_axis_angle(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: torch.Tensor | float = 0.0,
    grid_points: int = 1,
    iterations: int = 3,
    return_single_angle: bool = True,
) -> torch.Tensor:
    """Refine the tilt axis angle for electron tomography data.

    Uses common line projections and LBFGS optimization to find the optimal
    tilt axis angle(s) that minimize differences between projections across
    the tilt series.

    Parameters
    ----------
    tilt_series : torch.Tensor
        Tensor containing the tilt series images with shape [n_tilts, height, width].
    tilt_angles : torch.Tensor
        Tilt angles at which the projection images were collected.
    tilt_axis_angle : float, default=0.0
        Initial guess for the tilt axis angle in degrees.
    grid_points : int, default=1
        Number of control points for the cubic B-spline grid. When > 1, allows for
        non-constant tilt axis angle across the tilt series.
    iterations : int, default=3
        Number of LBFGS optimization iterations to perform.
    return_single_angle : bool, default=True
        Return a single value for the tilt axis angle instead of a tensor
        with one value per tilt.

    Returns
    -------
    torch.Tensor or float
        If grid_points=1: a single float with the optimized mean tilt axis angle.
        If grid_points>1: a tensor of optimized tilt axis angles for each tilt.

    Notes
    -----
    The function works by:
    1. Applying a B-spline representation to model the tilt axis angle
    2. Projecting images perpendicular to the tilt axis
    3. Comparing these projections across different tilts
    4. Minimizing differences between projections using LBFGS optimizer

    Common line projections are normalized and weighted according to the
    projected mask to emphasize regions of interest.
    """
    n_tilts = tilt_series.shape[0]
    image_shape = tilt_series.shape[-2:]
    device = tilt_series.device
    if tilt_angles is not torch.Tensor:
        tilt_angles = torch.from_numpy(tilt_angles).to(device)
    # tilt_series = tilt_series * alignment_mask
    #
    # # generate a weighting for the common line ROI by projecting the mask
    # mask_weights = project_2d_to_1d(
    #     alignment_mask,
    #     torch.eye(2, device=device),  # angle does not matter for circle
    # )
    # mask_weights = mask_weights / mask_weights.max()  # normalise to 0 and 1

    edge_taper = _common_line_taper(
        image_shape[0],
        image_shape[0] / 2 * .7,
        image_shape[0] / 2 * .3,
    )
    edge_taper = einops.rearrange(edge_taper, 'w -> 1 w').to(device)

    # optimize tilt axis angle
    tilt_axis_grid = CubicBSplineGrid1d(resolution=grid_points, n_channels=1)
    tilt_axis_grid.data = torch.tensor(
        [
            tilt_axis_angle,
        ]
        * grid_points,
        dtype=torch.float32,
        device=device,
    )
    tilt_axis_grid.to(device)
    interpolation_points = torch.linspace(0, 1, n_tilts, device=device)

    lbfgs = torch.optim.LBFGS(
        tilt_axis_grid.parameters(),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        # The common line is the projection perpendicular to the
        # tilt-axis, hence add 90 degrees to project along the x-axis
        pred_tilt_axis_angles = tilt_axis_grid(interpolation_points) + 90.0
        M = R(pred_tilt_axis_angles, yx=False)
        M = M[:, :2, :2]  # we only need the rotation matrix
        M = M.to(device)

        sample_area_mask = _tilt_sample_mask(
            image_shape=image_shape,
            tilt_axis_angles=pred_tilt_axis_angles,
            tilt_angles=tilt_angles,
            device=device,
        )
        tilt_series_masked = tilt_series * sample_area_mask

        projections = torch.cat(
            [  # indexing with [[i]] does not drop the dimension
                project_2d_to_1d(
                    tilt_series_masked[i], M[[i]]
                ) for i in range(n_tilts)
            ]
        )
        projections = projections - einops.reduce(
            projections, "tilt w -> tilt 1", reduction="mean"
        )
        projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        projections = projections * edge_taper
        # mask_weights = torch.cat(
        #     [  # indexing with [[i]] does not drop the dimension
        #         project_2d_to_1d(
        #             sample_area_mask[i], M[[i]]) for i in range(n_tilts)
        #     ]
        # )
        # mask_weights = (
        #         mask_weights /
        #         mask_weights.max(dim=(-1), keepdim=True).values
        # )
        # projections = projections / mask_weights
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(projections.detach().cpu().numpy())
        # viewer.add_image(mask_weights.detach().cpu().numpy())
        # # viewer.add_image(tilt_series_masked.detach().cpu().numpy())
        # napari.run()
        # projections = projections * mask_weights  # weight the common lines
        lbfgs.zero_grad()
        squared_differences = (
            projections - einops.rearrange(projections, "b d -> b 1 d")
        ) ** 2
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(projections.detach().cpu().numpy())
        # napari.run()
        loss = einops.reduce(squared_differences, "b1 b2 d -> 1", reduction="sum")
        loss.backward()
        return loss

    for _ in range(iterations):
        lbfgs.step(closure)

    tilt_axis_angles = tilt_axis_grid(interpolation_points).detach()

    if return_single_angle:
        return torch.mean(tilt_axis_angles)
    else:
        return tilt_axis_angles
