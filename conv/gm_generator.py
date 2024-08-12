import itertools
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from signals import SIGNALS_2D
from sklearn import mixture


def plot_results(ax, X, Y, means, covariances, title):
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y == i):
            continue
        ax.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    ax.set_xlim(0, 4.0 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_xticks(np.linspace(0, 4.0 * np.pi, num=9))  # Show x-axis ticks
    ax.set_yticks(np.linspace(-1.5, 1.5, num=7))  # Show y-axis ticks


np.random.seed(0)
x = np.linspace(0, 4 * np.pi, 1000)

# Create a figure with subplots
fig, axs = plt.subplots(4, 3, figsize=(18, 18))

gmm_params = {}

for i, (signal_name, signal_func) in enumerate(SIGNALS_2D.items()):
    row, col = divmod(i, 3)
    y = signal_func(x)
    X = np.column_stack((x, y))

    # Fit a Gaussian mixture with EM using ten components
    gmm = mixture.GaussianMixture(
        n_components=100, covariance_type="full", max_iter=10000
    ).fit(X)

    gmm_params[signal_name] = {
        "weights": gmm.weights_,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
    }

    # Plot original signal
    axs[row, col].plot(x, y, label="Original Signal")

    # Plot GMM results
    plot_results(
        axs[row, col], X, gmm.predict(X), gmm.means_, gmm.covariances_, signal_name
    )
    axs[row, col].legend()

# Adjust layout and save the figure as an image
plt.tight_layout()
image_path = "gm_fit_signals_2d.png"
plt.savefig(image_path)
plt.close(fig)

with open("gmm_params.pkl", "wb") as f:
    pickle.dump(gmm_params, f)
