import torch
import torch.nn as nn
from ipdb import set_trace

from gsplat import quat_scale_to_covar_preci


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim, activation="relu"):
        super(MLP, self).__init__()
        assert activation in ["relu", "tanh", "none"], activation
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = (
            nn.ReLU()
            if activation == "relu"
            else nn.Tanh() if activation == "tanh" else nn.Identity()
        )

    def forward(self, x):
        x = self.act(self.fc1(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.fc2(x)
        return x


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
        reduced_weights = weights.sum(dim=dim)
        reduced_means = (weights.unsqueeze(-1) * means).sum(
            dim=dim
        ) / reduced_weights.unsqueeze(-1)
        reduced_covars = (
            weights.unsqueeze(-1).unsqueeze(-1)
            * (
                covars
                + (means - reduced_means.unsqueeze(dim))[:, :, :, None, :]
                * (means - reduced_means.unsqueeze(dim))[:, :, :, :, None]
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
        mlp_depth: int,
        dim: int,
    ) -> None:
        super(GMConv, self).__init__()

        self.Nk = num_kernels
        self.Nc = num_components
        self.D = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weights = nn.Parameter(
            torch.ones(self.Nk, self.Nc, device=self.device)
        )  # [Nk, Nc]
        self.means = nn.Parameter(
            torch.zeros(self.Nk, self.Nc, self.D, device=self.device)
        )  # [Nk, Nc, D]
        self.scales = nn.Parameter(
            torch.full((self.Nk, self.Nc, self.D), 1e-4, device=self.device)
        )  # [Nk, Nc, D]
        self.quats = nn.Parameter(
            torch.randn(self.Nk, self.Nc, 4, device=self.device)
        )  # [Nk, Nc, 4]
        self.features = nn.Parameter(torch.randn(self.Nk, device=self.device))  # [Nk,]

        self.weights_mlp = MLP(self.Nk, self.Nk // 2, mlp_depth, 1, activation="relu")
        self.means_mlp = MLP(self.Nk, self.Nk // 2, mlp_depth, 1, activation="none")
        self.covars_mlp = MLP(self.Nk, self.Nk // 2, mlp_depth, 1, activation="none")
        # self.features_mlp = MLP(self.Nk, self.Nk // 2, mlp_depth, 1, activation="relu")

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

        kernel_covars, _ = quat_scale_to_covar_preci(
            self.quats.reshape(-1, 4),
            self.scales.reshape(-1, self.D),
            compute_preci=False,
        )  # [Nk * Nc, D, D]
        kernel_covars = kernel_covars.reshape(
            self.Nk, self.Nc, self.D, self.D
        )  # [Nk, Nc, D, D]
        conv_weights = (
            self.weights.unsqueeze(0) * weights[:, None, None]
        )  # [Ni, Nk, Nc]
        conv_means = (
            self.means.unsqueeze(0) + means[:, None, None, :]
        )  # [Ni, Nk, Nc, D]
        conv_covars = (
            kernel_covars.unsqueeze(0) + covars[:, None, None, :, :]
        )  # [Ni, Nk, Nc, D, D]
        reduced_weights, reduced_means, reduced_covars = _reduce_gaussians(
            conv_weights, conv_means, conv_covars, reduction="merge", dim=2
        )  # [Ni, Nk], [Ni, Nk, D], [Ni, Nk, D, D]

        # reduced_features = (
        #     features[:, None, :] * self.features[None, None, :]
        # )  # [Ni, Nk, Nf]
        weights_ = self.weights_mlp(reduced_weights)[:, 0]  # [Ni]
        means_ = self.means_mlp(reduced_means.permute(0, 2, 1))[:, :, 0]  # [Ni, D]
        covars_ = self.covars_mlp(reduced_covars.permute(0, 2, 3, 1))[
            :, :, :, 0
        ]  # [Ni, D, D]

        # features_ = self.features_mlp(reduced_features.permute(0, 2, 1))  # [Ni, Nf]
        features_ = features  # TODO: reduce features
        return weights_, means_, covars_, features_


if __name__ == "__main__":
    N = 1000
    D = 3
    Nf = 32
    Nk = 16
    Nc = 8
    mlp_depth = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = GMConv(Nk, Nc, mlp_depth, D).to(device)
    weights = torch.randn(N, device=device)
    means = torch.randn(N, D, device=device)
    covars = torch.randn(N, D, D, device=device)
    features = torch.randn(N, Nf, device=device)
    weights_, means_, covars_, features_ = conv(weights, means, covars, features)
    print(weights_.shape, means_.shape, covars_.shape, features_.shape)
