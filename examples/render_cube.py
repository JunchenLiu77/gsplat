"""
Generate a cube dataset.
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import _rasterization, rasterization
from datasets.traj import generate_generic_spiral_path


@torch.no_grad()
def render_traj(
    result_dir: str,
    width: int,
    height: int,
    device: torch.device,
    focal: float,
    radii: float,
):
    """Entry for trajectory rendering."""
    print("Running trajectory rendering...")

    camtoworlds_all = generate_generic_spiral_path(
        n_frames=120,
        radii=radii,
    )
    camtoworlds_all = np.concatenate(
        [
            camtoworlds_all,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0),
        ],
        axis=1,
    )  # [N, 4, 4]

    camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
    K = torch.eye(3, device=device)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2
    K[1, 2] = height / 2

    # save to video
    video_dir = f"{result_dir}/videos"
    os.makedirs(video_dir, exist_ok=True)
    writer = imageio.get_writer(f"{video_dir}/traj.mp4", fps=30)
    for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
        camtoworlds = camtoworlds_all[i : i + 1]
        Ks = K[None]

        # ----- render a cube -----
        N = 5
        means = torch.zeros((N, 3), device=device)
        quats = torch.ones((N, 4), device=device)
        scales = torch.full((N, 3), 1.0, device=device) * radii / 10
        opacities = torch.full((N,), 1.0, device=device)
        colors = torch.ones(1, 3, device=device).repeat(N, 1)
        tquats = None
        tscales = None

        verts = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device=device,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 4],
                [1, 2, 3, 7],
                [1, 2, 4, 7],
                [1, 4, 5, 7],
                [2, 4, 6, 7],
            ],
            device=device,
        )
        tvertices = verts[indices] * radii / 10

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            sparse_grad=False,
            rasterize_mode="RGB",
            distributed=False,
            camera_model="pinhole",
            enable_culling=True,
            tscales=tscales,
            tquats=tquats,
            tvertices=tvertices,
        )
        colors = torch.clamp(render_colors[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
        canvas_list = [colors]

        # write images
        canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
        canvas = (canvas * 255).astype(np.uint8)
        writer.append_data(canvas)
    writer.close()
    print(f"Video saved to {video_dir}/traj.mp4")


if __name__ == "__main__":
    render_traj(
        result_dir="results/cube",
        width=400,
        height=300,
        device=torch.device("cuda"),
        focal=1000.0,
        radii=10.0,
    )
