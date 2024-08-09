import torch
import torch.nn as nn

from gsplat import quat_scale_to_covar_preci


def _sample(
    weights: torch.Tensor,
    means: torch.Tensor,
    covars: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Sample n most important points from a Gaussian Mixture.

    Args:
        n (int): Number of points to sample
        weights (torch.Tensor): Mixture weights
        means (torch.Tensor): Mixture means
        covars (torch.Tensor): Mixture covariances

    Returns:
        torch.Tensor: Sampled points
    """
    N, D = means.shape
    assert weights.shape == (N,), weights.shape
    assert covars.shape == (N, D, D), covars.shape

    # TODO: sample from the mixture
    raise NotImplementedError
    return samples  # [Ns, D]


def _gaussian_bound(
    means: torch.Tensor,
    covars: torch.Tensor,
) -> torch.Tensor:
    """Compute the AABB of a Gaussian distribution.

    Args:
        means (torch.Tensor): Mixture means
        covars (torch.Tensor): Mixture covariances

    Returns:
        torch.Tensor: Min and max bounds
    """
    N, D = means.shape
    assert covars.shape == (N, D, D), covars.shape

    raise NotImplementedError
    return mins, maxs  # [N, D], [N, D]


def _detect_intersected(
    mins0: torch.Tensor,
    maxs0: torch.Tensor,
    mins1: torch.Tensor,
    maxs1: torch.Tensor,
) -> torch.Tensor:
    """Detect intersected components based on the AABB.

    Args:
        mins0 (torch.Tensor): Min bounds of the first AABB
        maxs0 (torch.Tensor): Max bounds of the first AABB
        mins1 (torch.Tensor): Min bounds of the second AABB
        maxs1 (torch.Tensor): Max bounds of the second AABB

    Returns:
        torch.Tensor: Intersected components
    """
    assert mins0.shape == maxs0.shape, (mins0.shape, maxs0.shape)
    assert mins1.shape == maxs1.shape, (mins1.shape, maxs1.shape)

    # TODO: more efficient implementation
    intersected = torch.ones(mins0.shape[0], mins1.shape[0], dtype=torch.bool)
    return intersected  # [N0, N1]


class GMConv(nn.Module):
    """Gaussian Mixture Convolutional Layer.
    Each convolutional kernel is represented with a Gaussian Mixture.

    Args:
        in_components (int): Number of input mixture components
        out_components (int): Number of output mixture components
        dim (int): Dimension of the gaussian
        kernel_components (int): Number of kernel mixture components
    """

    # TODO: maybe change components to shorter names
    def __init__(
        self,
        in_components: int,
        out_components: int,
        kernel_components: int,
        dim: int,
    ) -> None:
        super(GMConv, self).__init__()

        self.Ni = in_components
        self.No = out_components
        self.Nk = kernel_components
        self.D = dim

        self.weights = nn.Parameter(torch.randn(self.No, self.Nk))  # [No, Nk]
        self.means = nn.Parameter(torch.randn(self.No, self.Nk, self.D))  # [No, Nk, D]
        self.scales = nn.Parameter(torch.randn(self.No, self.Nk, self.D))  # [No, Nk, D]
        self.quats = nn.Parameter(torch.randn(self.No, self.Nk, 4))  # [No, Nk, 4]

    def forward(
        self,
        weights: torch.Tensor,  # [Ni]
        means: torch.Tensor,  # [Ni, D]
        covars: torch.Tensor,  # [Ni, D, D]
        features: torch.Tensor,  # [Ni, Nf]
    ) -> torch.Tensor:
        assert weights.shape == (self.Ni,), weights.shape
        assert means.shape == (self.Ni, self.D), means.shape
        assert covars.shape == (self.Ni, self.D, self.D), covars.shape
        assert features.shape[0] == self.Ni, features.shape

        # Step1: Sample n points based on the input mixture density.
        kernel_means = _sample(weights, means, covars, self.No)  # [No, D]
        component_weights = self.weights  # [No, Nk]
        component_means = kernel_means.unsqueeze(1) + self.means  # [No, Nk, D]
        component_covars, _ = quat_scale_to_covar_preci(
            self.quats.reshape(-1, 4),
            self.scales.reshape(-1, self.D),
            compute_preci=False,
        )  # [No * Nk, D, D]
        component_covars = component_covars.reshape(
            self.No, self.Nk, self.D, self.D
        )  # [No, Nk, D, D]

        # # Step2: Define the Neighbourhood for each kernel component.
        # # We use AABB to define the neighbourhood.
        # component_mins, component_maxs = _gaussian_bound(
        #     component_means.reshape(-1, self.D),
        #     component_covars.reshape(-1, self.D, self.D),
        # )  # [No * Nk, D], [No * Nk, D]
        # component_mins = component_mins.reshape(self.No, self.Nk, self.D)   # [No, Nk, D]
        # component_maxs = component_maxs.reshape(self.No, self.Nk, self.D)   # [No, Nk, D]
        # input_mins, input_maxs = _gaussian_bound(means, covars)  # [Ni, D], [Ni, D]

        # # Step3: Determine the intersected components based on the neighbourhood.
        # intersected = _detect_intersected(
        #     input_mins, input_maxs, component_mins, component_maxs
        # )  # [No, Nk, Ni]

        # Step4: Convolve the input features.
        # TODO: change here to use only the intersected components

        convolved_weights = (
            component_weights.unsqueeze(2) * weights[None, None, :]
        )  # [No, Nk, Ni]
        convolved_means = (
            component_means.unsqueeze(2) + means[None, None, :]
        )  # [No, Nk, Ni, D]
        convolved_covars = (
            component_covars.unsqueeze(2) + covars[None, None, :]
        )  # [No, Nk, Ni, D, D]

        kernel_weights, kernel_means, kernel_covars = _merge_gaussians(
            convolved_weights, convolved_means, convolved_covars, dim=2
        )  # [No, Nk], [No, Nk, D], [No, Nk, D, D]
