import torch
import torch.nn as nn
from ipdb import set_trace

from gsplat import quat_scale_to_covar_preci


def _reduce_gaussians(
    weights: torch.Tensor,  # [Nk, Nc, Ni]
    means: torch.Tensor,  # [Nk, Nc, Ni, D]
    covars: torch.Tensor,  # [Nk, Nc, Ni, D, D]
    reduction: str,
    dim: int,
) -> torch.Tensor:
    """Reduce the Gaussian Mixture based on the reduction method.

    Args:
        weights (torch.Tensor): Mixture weights
        means (torch.Tensor): Mixture means
        covars (torch.Tensor): Mixture covariances
        reduction (str): Reduction method
        dim (int): Dimension to reduce

    Returns:
        torch.Tensor: Reduced Gaussian Mixture
    """
    Nk, Nc, Ni, D = means.shape
    assert weights.shape == (Nk, Nc, Ni), weights.shape
    assert covars.shape == (Nk, Nc, Ni, D, D), covars.shape
    assert reduction in ["merge", "max"], reduction

    if reduction == "max":
        indices = torch.argmax(weights, dim=dim)  # [Nk, Ni]

        set_trace()
        reduced_weights = torch.gather(weights, dim=dim, index=indices.unsqueeze(dim))
        reduced_means = torch.gather(
            means,
            dim=dim,
            index=indices.unsqueeze(dim).unsqueeze(-1).expand(-1, -1, -1, D),
        )
        reduced_covars = torch.gather(
            covars,
            dim=dim,
            index=indices.unsqueeze(dim)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, D, D),
        )
    else:
        set_trace()
        reduced_weights = weights.sum(dim=dim)
        reduced_means = (weights.unsqueeze(-1) * means).sum(
            dim=dim
        ) / reduced_weights.unsqueeze(-1)
        reduced_covars = (
            weights.unsqueeze(-1).unsqueeze(-1)
            * (
                covars
                + (means.unsqueeze(-1) - reduced_means.unsqueeze(-2)).unsqueeze(-1)
                @ (means.unsqueeze(-1) - reduced_means.unsqueeze(-2)).unsqueeze(-2)
            )
        ).sum(dim=dim) / reduced_weights.unsqueeze(-1).unsqueeze(-1)
    return reduced_weights, reduced_means, reduced_covars


class GMConv(nn.Module):
    """Gaussian Mixture Convolutional Layer.
    Each convolutional kernel is represented with a Gaussian Mixture.

    Args:
        dim (int): Dimension of the gaussian
        num_kernels (int): Number of kernels
        num_gms (int): Number of Gaussian Mixtures in each kernel
    """

    # TODO: maybe change components to shorter names
    def __init__(
        self,
        num_kernels: int,
        num_components: int,
        dim: int,
    ) -> None:
        super(GMConv, self).__init__()

        self.Nk = num_kernels
        self.Nc = num_components
        self.D = dim

        self.weights = nn.Parameter(torch.randn(self.Nk, self.Nc))  # [Nk, Nc]
        self.means = nn.Parameter(torch.randn(self.Nk, self.Nc, self.D))  # [Nk, Nc, D]
        self.scales = nn.Parameter(torch.randn(self.Nk, self.Nc, self.D))  # [Nk, Nc, D]
        self.quats = nn.Parameter(torch.randn(self.Nk, self.Nc, 4))  # [Nk, Nc, 4]

    def forward(
        self,
        weights: torch.Tensor,  # [Ni]
        means: torch.Tensor,  # [Ni, D]
        covars: torch.Tensor,  # [Ni, D, D]
        features: torch.Tensor,  # [Ni, Nf]
    ) -> torch.Tensor:

        Ni, Nf = features.shape
        assert weights.shape == (Ni,), weights.shape
        assert means.shape == (Ni, self.D), means.shape
        assert covars.shape == (Ni, self.D, self.D), covars.shape

        # Step1: Sample n points based on the input mixture density.
        kernel_covars, _ = quat_scale_to_covar_preci(
            self.quats.reshape(-1, 4),
            self.scales.reshape(-1, self.D),
            compute_preci=False,
        )  # [Nk * Nc, D, D]
        kernel_covars = kernel_covars.reshape(
            self.Nk, self.Nc, self.D, self.D
        )  # [Nk, Nc, D, D]

        # Step4: Convolve the input features.
        # TODO: change here to use only the intersected components

        conv_weights = (
            self.weights.unsqueeze(2) * weights[None, None, :]
        )  # [Nk, Nc, Ni]
        conv_means = self.means.unsqueeze(2) + means[None, None, :]  # [Nk, Nc, Ni, D]
        conv_covars = (
            kernel_covars.unsqueeze(2) + covars[None, None, :]
        )  # [Nk, Nc, Ni, D, D]

        reduced_weights, reduced_means, reduced_covars = _reduce_gaussians(
            conv_weights, conv_means, conv_covars, reduction="merge", dim=1
        )  # [Nk, Ni], [Nk, Ni, D], [Nk, Ni, D, D]
