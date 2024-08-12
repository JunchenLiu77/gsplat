import itertools
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from gm_conv import GMConv
from ipdb import set_trace
from signals import SIGNALS_2D


def _plot_gmm_results(ax, X, weights, means, covars, iter_num):
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

    with torch.no_grad():
        for i, (weight, mean, covar, color) in enumerate(
            zip(weights, means, covars, color_iter)
        ):
            mean = mean.cpu().numpy()
            covar = covar.cpu().numpy()

            v, w = np.linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])

            # Plot data points (assuming we have labels, if not, adjust accordingly)
            ax.scatter(X[:, 0], X[:, 1], 0.8)

            # Plot ellipse
            angle = np.arctan2(u[1], u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(
                mean,
                v[0],
                v[1],
                angle=180.0 + angle,
                color=color,
                alpha=0.5,
            )
            ell.set_clip_box(ax.bbox)
            ax.add_artist(ell)

    # Set plot limits and labels
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_title(f"Iteration {iter_num}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Add grid and ticks
    ax.grid(True)
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=9))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=7))


def _eval_gmm(weights, means, covars, x):
    """
    Evaluate the Gaussian Mixture Model on the input data x.

    Args:
        weights (torch.Tensor): Tensor of shape (N,) representing the weights of the GMM components.
        means (torch.Tensor): Tensor of shape (N, D) representing the means of the GMM components.
        covars (torch.Tensor): Tensor of shape (N, D, D) representing the covariance matrices of the GMM components.
        x (torch.Tensor): Tensor of shape (M, D) representing the data points.

    Returns:
        torch.Tensor: The evaluation of the GMM on the input data, of shape (M,).
    """
    N, D = means.shape
    M = x.shape[0]

    assert weights.shape == (N,), weights.shape
    assert covars.shape == (N, D, D), covars.shape
    assert x.shape[1] == D, x.shape

    # Calculate normalization constant for Gaussian
    norm_const = (2 * torch.pi) ** (-0.5 * D)

    # Compute the Cholesky decomposition for numerical stability
    L = torch.linalg.cholesky(covars)
    inv_covar = torch.cholesky_inverse(L)
    det_covar = torch.det(covars)

    # Compute differences
    x_expanded = x.unsqueeze(1)  # (M, 1, D)
    means_expanded = means.unsqueeze(0)  # (1, N, D)
    diff = x_expanded - means_expanded  # (M, N, D)

    # Calculate Mahalanobis distance
    inv_covar_expanded = inv_covar.unsqueeze(0)  # (1, N, D, D)
    diff = diff.unsqueeze(-1)  # (M, N, D, 1)
    mahalanobis_dist = torch.matmul(inv_covar_expanded, diff).squeeze(-1)  # (M, N, D)
    exp_term = torch.sum(diff.squeeze(-1) * mahalanobis_dist, dim=-1)  # (M, N)

    # Calculate PDF values
    exp_term = torch.exp(-0.5 * exp_term)  # (M, N)
    component_pdf = (weights / torch.sqrt(det_covar)) * exp_term  # (N,)
    component_pdf = component_pdf * norm_const  # (M, N)

    # Sum across components for final GMM evaluation
    gmm_eval = component_pdf.sum(dim=1)  # (M,)
    return gmm_eval


def convert_signals_2d(src_signal, tar_signal, params_path, device, model):
    assert src_signal in SIGNALS_2D, src_signal
    assert tar_signal in SIGNALS_2D, tar_signal

    # Load GMM parameters
    with open(params_path, "rb") as f:
        gmm_params = pickle.load(f)

    src_weights = torch.tensor(
        gmm_params[src_signal]["weights"], dtype=torch.float, device=device
    )
    src_means = torch.tensor(
        gmm_params[src_signal]["means"], dtype=torch.float, device=device
    )
    src_covars = torch.tensor(
        gmm_params[src_signal]["covariances"], dtype=torch.float, device=device
    )

    tar_x = torch.tensor(gmm_params[tar_signal]["x"], dtype=torch.float, device=device)
    tar_y = torch.tensor(gmm_params[tar_signal]["y"], dtype=torch.float, device=device)
    X = torch.stack((tar_x, tar_y), dim=1).cpu().numpy()

    # Convert the source signal to the target signal
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for iter in range(10000):
        input = {
            "weights": src_weights,
            "means": src_means,
            "covars": src_covars,
            "features": torch.rand(src_weights.shape[0], 1, device=device),
        }
        output = model(input)
        weights, means, covars = output["weights"], output["means"], output["covars"]
        weights = torch.softmax(weights, dim=0)

        # calculate the probability of the target signal
        prob = _eval_gmm(weights, means, covars, torch.stack((tar_x, tar_y), dim=1))
        loss = -torch.log(prob.clamp(1e-10)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Iter: {iter}, Loss: {loss.item()}")

        # plot the target signal and current GM
        if iter % 100 == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            _plot_gmm_results(ax, X, weights, means, covars, iter)
            plt.savefig(f"gm_from_{src_signal}_to_{tar_signal}.png")
            plt.close(fig)
    set_trace()


if __name__ == "__main__":
    D = 2
    Nf = 1
    Nk = 32
    Nc = 16
    mlp_depth = 4
    num_conv_layers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_path = "gmm_params.pkl"

    layers = []
    for i in range(num_conv_layers):
        layers.append(GMConv(Nk, Nc, mlp_depth, D))
    model = torch.nn.Sequential(*layers).to(device)
    # model = GMConv(Nk, Nc, mlp_depth, D).to(device)
    convert_signals_2d(
        src_signal="sine_wave",
        tar_signal="fm_signal",
        params_path=params_path,
        device=device,
        model=model,
    )
