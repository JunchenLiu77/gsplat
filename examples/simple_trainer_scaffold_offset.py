import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from torch.nn import ModuleDict, ParameterDict
from torch.optim import SparseAdam, Adam

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
from utils import CameraOptModule, knn, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import ScaffoldStrategy


class PositionalEncodingMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int,
        num_frequencies: int,
        use_residual: bool,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.use_residual = use_residual

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    torch.nn.Linear(input_dim * num_frequencies * 2, latent_dim)
                )
            else:
                layers.append(torch.nn.Linear(latent_dim, latent_dim))
            layers.append(torch.nn.BatchNorm1d(latent_dim))
            layers.append(torch.nn.ReLU(True))
            if use_residual and i > 0:
                layers.append(ResidualConnection(latent_dim))
        layers.append(torch.nn.Linear(latent_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequencies = 2.0 ** torch.arange(
            self.num_frequencies, dtype=torch.float32, device=x.device
        )
        x = x.unsqueeze(-1)
        encoded = (x * frequencies).view(x.shape[0], -1)
        x = torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)
        return self.mlp(x)


class ResidualConnection(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x + torch.nn.functional.relu(x)


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Render trajectory path
    render_traj_path: str = "ellipse"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "examples/data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Dimensionality of anchor features
    feat_dim: int = 256
    # Number offsets
    n_feat_offsets: int = 4

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # voxel size for Scaffold-GS
    voxel_size = 0.001
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Turn on another SH degree every this steps
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: ScaffoldStrategy = field(default_factory=ScaffoldStrategy)
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Scale regularization
    scale_reg: float = 0.1
    # Opacity regularization
    opacity_reg: float = 0.1
    # Offset regularization
    offset_reg: float = 0.1

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)

        strategy = self.strategy
        strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
        strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
        strategy.voxel_size = self.voxel_size


def create_splats_with_optimizers(
    parser: Parser,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    voxel_size: float = 0.001,
) -> tuple[
    dict[str, ModuleDict | ParameterDict], dict[str, dict[str, SparseAdam | Adam]]
]:

    # Compare GS-Scaffold paper formula (4)
    # points = np.unique(np.round(parser.points / voxel_size), axis=0) * voxel_size
    points = torch.from_numpy(parser.points).float()
    colors = torch.from_numpy(parser.points_rgb / 255.0).float().cuda()

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size].cuda()
    scales = scales[world_rank::world_size].cuda()
    N = points.shape[0]

    features = torch.zeros((N, cfg.feat_dim))
    offsets = torch.randn((N, cfg.n_feat_offsets, 3)) * scene_scale * 0.01

    opacities = torch.logit(torch.full((N, 1), init_opacity)).cuda()  # [N,]

    # Define learning rates for gauss_params and decoders
    learning_rates = {
        "anchors": 0,  # 1e-4
        # "features": 7.5e-3,
        # "scales": 7e-3,
        # "opacities": 5e-2,
        "offsets": 1e-3,
        "feature_mlp": 1e-3,
        # "opacities_mlp": 2e-3,
        # "colors_mlp": 1e-2,
        # "scale_rot_mlp": 4e-4,
    }

    # Define gauss_params
    gauss_params = torch.nn.ParameterDict(
        {
            "anchors": torch.nn.Parameter(points),
            # "features": torch.nn.Parameter(features),
            # "scales": torch.nn.Parameter(scales),
            # "opacities": torch.nn.Parameter(opacities),
            "offsets": torch.nn.Parameter(offsets),
        }
    ).to(device)

    feature_mlp: torch.nn.Sequential = PositionalEncodingMLP(
        input_dim=3,
        latent_dim=cfg.feat_dim,
        output_dim=11,  # color + opacity + quat + scale [3 + 1 + 4 + 3]
        num_layers=8,
        num_frequencies=10,
        use_residual=True,
    ).cuda()

    # # Define the MLPs (decoders)
    # colors_mlp: torch.nn.Sequential = torch.nn.Sequential(
    #     torch.nn.Linear(cfg.feat_dim, cfg.feat_dim),
    #     torch.nn.ReLU(True),
    #     torch.nn.Linear(cfg.feat_dim, cfg.feat_dim),
    #     torch.nn.ReLU(True),
    #     torch.nn.Linear(cfg.feat_dim, cfg.feat_dim),
    #     torch.nn.ReLU(True),
    #     torch.nn.Linear(cfg.feat_dim, 3 * cfg.n_feat_offsets),
    #     torch.nn.Sigmoid(),
    # ).cuda()

    # opacities_mlp: torch.nn.Sequential = torch.nn.Sequential(
    #     torch.nn.Linear(cfg.feat_dim, cfg.feat_dim),
    #     torch.nn.ReLU(True),
    #     torch.nn.Linear(cfg.feat_dim, cfg.n_feat_offsets),
    #     torch.nn.Tanh(),
    # ).cuda()

    # scale_rot_mlp: torch.nn.Sequential = torch.nn.Sequential(
    #     torch.nn.Linear(cfg.feat_dim, cfg.feat_dim),
    #     torch.nn.ReLU(True),
    #     torch.nn.Linear(cfg.feat_dim, 7 * cfg.n_feat_offsets),
    # ).cuda()

    # Initialize decoders (MLPs)
    decoders = torch.nn.ModuleDict(
        {
            "feature_mlp": feature_mlp,
            # "opacities_mlp": opacities_mlp,
            # "colors_mlp": colors_mlp,
            # "scale_rot_mlp": scale_rot_mlp,
        }
    ).to(device)

    # Scale learning rates based on batch size (BS)
    BS = batch_size * world_size

    # Create optimizers for gauss_params
    gauss_optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": param, "lr": learning_rates[name] * math.sqrt(BS)}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, param in gauss_params.items()
    }

    # Create optimizers for decoders
    decoders_optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [
                {
                    "params": decoder.parameters(),
                    "lr": learning_rates[name] * math.sqrt(BS),
                }
            ],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, decoder in decoders.items()
    }

    # Combine gauss_params and decoders optimizers into a dictionary of dictionaries
    optimizers = {
        "gauss_optimizer": gauss_optimizers,
        "decoders_optimizer": decoders_optimizers,
    }

    ############################################################################################
    # TODO: fit the mlp to make opacities and scales small at first
    for i in range(100):
        x = torch.randn(N, 3).cuda()
        y = feature_mlp(x)
        colors_, opacities_, quats_, scales_ = y.split([3, 1, 4, 3], dim=-1)
        loss = (
            torch.nn.functional.mse_loss(opacities_, opacities)
            + torch.nn.functional.mse_loss(scales_, scales)
            + torch.nn.functional.mse_loss(colors_, colors)
        )
        print(f"[warmup] iter {i}: loss = {loss.item()}")
        optimizers["decoders_optimizer"]["feature_mlp"].zero_grad()
        loss.backward()
        optimizers["decoders_optimizer"]["feature_mlp"].step()

    print(
        f"[warmup] done, avg opacities = {opacities.sigmoid().mean().item()}, avg scales = {scales.exp().mean().item()}"
    )
    optimizers["decoders_optimizer"]["feature_mlp"] = torch.optim.Adam(
        optimizers["decoders_optimizer"]["feature_mlp"].param_groups,
        lr=1e-4 * math.sqrt(BS),
        eps=1e-15 / math.sqrt(BS),
        betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
    )
    ############################################################################################

    # Return the gauss_params, decoders, and the dictionary of dictionaries for optimizers
    splats = {"gauss_params": gauss_params, "decoders": decoders}
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.cfg.strategy.voxel_size = self.cfg.voxel_size
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            voxel_size=cfg.voxel_size,
        )
        print(
            "Model initialized. Number of GS:",
            len(self.splats["gauss_params"]["anchors"]),
        )

        # Densification Strategy
        self.cfg.strategy.check_sanity(
            self.splats["gauss_params"], self.optimizers["gauss_optimizer"]
        )

        self.strategy_state = self.cfg.strategy.initialize_state(
            scene_scale=self.scene_scale,
            feat_dim=cfg.feat_dim,
            n_feat_offsets=cfg.n_feat_offsets,
        )

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:

        # Get all the gaussians per voxel spawned from the anchors
        means = (
            self.splats["gauss_params"]["anchors"][:, None, :]
            + self.splats["gauss_params"]["offsets"]
        ).view(-1, 3)
        # means = self.splats["gauss_params"]["anchors"]
        features = self.splats["decoders"]["feature_mlp"](means)
        vis_colors, vis_opacity, quats, scales = features.split([3, 1, 4, 3], dim=-1)

        info = {
            "means": means,
            "colors": vis_colors.sigmoid(),
            "opacities": vis_opacity.sigmoid()[:, 0],
            "scales": scales.exp(),
            "quats": quats / quats.norm(dim=-1, keepdim=True),
            "offsets": self.splats["gauss_params"]["offsets"],
            # "neural_opacities": neural_opacity,
            # "neural_selection_mask": all_neural_gaussians,
            # "visible_anchor_mask": visible_anchor_mask,
        }

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, raster_info = rasterization(
            means=info["means"],
            quats=info["quats"],
            scales=info["scales"],
            opacities=info["opacities"],
            colors=info["colors"],
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.strategy.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            **kwargs,
        )
        raster_info.update(info)
        return render_colors, render_alphas, raster_info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["gauss_optimizer"]["anchors"],
                gamma=0.001 ** (1.0 / max_steps),
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["decoders_optimizer"]["feature_mlp"],
                gamma=0.001 ** (1.0 / max_steps),
            ),
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizers["gauss_optimizer"]["offsets"],
            #     gamma=(0.01 * self.scene_scale) ** (1.0 / max_steps),
            # ),
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizers["decoders_optimizer"]["opacities_mlp"],
            #     gamma=0.001 ** (1.0 / max_steps),
            # ),
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizers["decoders_optimizer"]["colors_mlp"],
            #     gamma=0.00625 ** (1.0 / max_steps),
            # ),
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizers["decoders_optimizer"]["scale_rot_mlp"], gamma=1.0
            # ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

        # self.splats["decoders"]["scale_rot_mlp"].train()
        # self.splats["decoders"]["opacities_mlp"].train()
        # self.splats["decoders"]["colors_mlp"].train()
        self.splats["decoders"]["feature_mlp"].train()

        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats["gauss_params"],
                optimizers=self.optimizers["gauss_optimizer"],
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda)
            loss += ssimloss * cfg.ssim_lambda
            desc = f"loss={loss.item():.3f}| "
            if cfg.scale_reg > 0:
                scale_loss = info["scales"].prod(dim=1).mean() * cfg.scale_reg
                loss += scale_loss
                desc += f"scale loss={scale_loss.item():.6f}| "
            if cfg.opacity_reg > 0:
                opacity_loss = info["opacities"].mean() * cfg.opacity_reg
                loss += opacity_loss
                desc += f"opacity loss={opacity_loss.item():.6f}| "
            if cfg.offset_reg > 0:
                offset_loss = info["offsets"].norm(dim=-1).mean() * cfg.offset_reg
                loss += offset_loss
                desc += f"offset loss={offset_loss.item():.6f}| "

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            loss.backward()

            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["gauss_params"]["anchors"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["gauss_params"]["anchors"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {
                    "step": step,
                    "feat_dim": self.cfg.feat_dim,
                    "n_feat_offsets": self.cfg.n_feat_offsets,
                    "gauss_params": self.splats["gauss_params"].state_dict(),
                    "opacities_mlp": self.splats["decoders"][
                        "opacities_mlp"
                    ].state_dict(),
                    "colors_mlp": self.splats["decoders"]["colors_mlp"].state_dict(),
                    "scale_rot_mlp": self.splats["decoders"][
                        "scale_rot_mlp"
                    ].state_dict(),
                }
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # For now no post steps
            self.cfg.strategy.step_post_backward(
                params=self.splats["gauss_params"],
                optimizers=self.optimizers["gauss_optimizer"],
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats["gauss_params"].keys():
                    grad = self.splats["gauss_params"][k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats["gauss_params"][k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats["gauss_params"][k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers["gauss_optimizer"].values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # optimize
            for optimizer in self.optimizers["decoders_optimizer"].values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(
                    step,
                    n_feat_offsets=self.cfg.n_feat_offsets,
                    feat_dim=self.cfg.feat_dim,
                )
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, n_feat_offsets: int, feat_dim: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        assert (
            n_feat_offsets == self.cfg.n_feat_offsets
        ), f"Feature offset count changed, should be {n_feat_offsets}"
        assert (
            feat_dim == self.cfg.feat_dim
        ), f"Feature dim changed, should be {feat_dim}"

        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["gauss_params"]["anchors"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=24)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=None,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]

        for k in runner.splats["gauss_params"].keys():
            runner.splats["gauss_params"][k].data = torch.cat(
                [ckpt["gauss_params"][k] for ckpt in ckpts]
            )
        for k in runner.splats["decoders"].keys():
            runner.splats["decoders"][k].load_state_dict(ckpts[0][k])
        step = ckpts[0]["step"]
        n_feat_offsets = ckpts[0]["n_feat_offsets"]
        feat_dim = ckpts[0]["feat_dim"]
        runner.eval(step=step, n_feat_offsets=n_feat_offsets, feat_dim=feat_dim)
        runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
