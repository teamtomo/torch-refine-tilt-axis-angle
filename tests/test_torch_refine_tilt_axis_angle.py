import pytest
import torch
from unittest.mock import patch

from torch_refine_tilt_axis_angle import refine_tilt_axis_angle


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a small tilt series (5 images of 20x20 pixels)
    tilt_series = torch.rand(5, 20, 20, device=device)

    # Create a simple circular mask
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 20, device=device),
        torch.linspace(-1, 1, 20, device=device),
        indexing='ij'
    )
    mask = ((x ** 2 + y ** 2) < 0.8 ** 2).float()

    return {
        'tilt_series': tilt_series,
        'mask': mask,
        'device': device
    }


def test_refine_tilt_axis_angle_default_parameters(sample_data):
    """Test refine_tilt_axis_angle with default parameters."""
    with patch('torch.optim.LBFGS.step'):
        # Setup
        tilt_series = sample_data['tilt_series']
        mask = sample_data['mask']

        # Call the function with default parameters
        result = refine_tilt_axis_angle(
            tilt_series=tilt_series,
            alignment_mask=mask
        )

        # Assertions
        assert isinstance(
            result, float
        )  # With grid_points=1, should return float


def test_refine_tilt_axis_angle_with_single_grid_point(sample_data):
    """Test refine_tilt_axis_angle with a single grid point."""
    with patch('torch.optim.LBFGS.step'):
        # Setup
        tilt_series = sample_data['tilt_series']
        mask = sample_data['mask']
        initial_angle = 45.0

        # Call the function
        result = refine_tilt_axis_angle(
            tilt_series=tilt_series,
            alignment_mask=mask,
            initial_tilt_axis_angle=initial_angle,
            grid_points=1,
            iterations=2
        )

        # Assertions
        assert isinstance(
            result, float
        )  # With grid_points=1, should return float


def test_refine_tilt_axis_angle_with_multiple_grid_points(sample_data):
    """Test refine_tilt_axis_angle with multiple grid points."""
    with patch('torch.optim.LBFGS.step'):
        # Setup
        tilt_series = sample_data['tilt_series']
        mask = sample_data['mask']
        initial_angle = 45.0

        # Call the function
        result = refine_tilt_axis_angle(
            tilt_series=tilt_series,
            alignment_mask=mask,
            initial_tilt_axis_angle=initial_angle,
            grid_points=3,
            iterations=2
        )

        # Assertions
        assert isinstance(result,
                          torch.Tensor)  # With grid_points>1, should return tensor
        assert result.shape[0] == tilt_series.shape[0]  # One angle per tilt
        assert result.requires_grad == False  # Should be detached


def test_refine_tilt_axis_angle_optimization_iterations(sample_data):
    """Test that the function performs the specified number of optimization iterations."""
    # Setup
    tilt_series = sample_data['tilt_series']
    mask = sample_data['mask']
    initial_angle = 45.0

    # Call the function with different iteration counts
    with patch('torch.optim.LBFGS.step') as mock_step:
        refine_tilt_axis_angle(
            tilt_series=tilt_series,
            alignment_mask=mask,
            initial_tilt_axis_angle=initial_angle,
            grid_points=3,
            iterations=5
        )

        # Assert that step was called the correct number of times
        assert mock_step.call_count == 5