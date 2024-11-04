import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable

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
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
from utils import set_random_seed

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from pykeops.torch import generic_argkmin

knn1 = generic_argkmin(
    "SqDist(x, y)",
    "a = Vi(1)",
    "x = Vi(3)",
    "y = Vj(3)",
)


def asym_chamfer(pc1, pc2):
    # encourage pc2 to approach pc1
    nn_indices = knn1(pc2, pc1)  # [N, 4]
    return (pc2[:, None, :] - pc1[nn_indices]).norm(dim=-1).mean()


class PositionalEncoding(torch.nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, num_bases, include_input):
        super(PositionalEncoding, self).__init__()
        self.num_bases = num_bases
        self.include_input = include_input
        self.output_dim = 6 * num_bases + (3 if include_input else 0)

        frequencies = 2.0 ** torch.arange(num_bases, dtype=torch.float32)
        phase_shifts = torch.tensor([0.0, math.pi / 2], dtype=torch.float32)

        self.register_buffer("frequencies", frequencies)
        self.register_buffer("phase_shifts", phase_shifts)

    def forward(
        self,
        x: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:
        orig_shape = x.shape  # [..., 3]
        x = x[..., None, None]  # [..., 3, 1, 1]
        frequencies = self.frequencies.view(1, 1, -1, 1)  # [1, 1, num_bases, 1]
        phase_shifts = self.phase_shifts.view(1, 1, 1, -1)  # [1, 1, 1, 2]
        args = x * frequencies * math.pi + phase_shifts  # [..., 3, num_bases, 2]
        encodings = torch.sin(args)  # [..., 3, num_bases, 2]
        encodings = encodings.reshape(*orig_shape[:-1], 6 * self.num_bases)

        if self.include_input:
            x_orig = x.squeeze(-1).squeeze(-1)  # [..., 3]
            encodings = torch.cat([x_orig, encodings], dim=-1)  # [..., output_dim]
        return encodings


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_init: Callable = torch.nn.init.kaiming_normal_,
        output_init: Callable = torch.nn.init.kaiming_normal_,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_init = hidden_init
        self.output_init = output_init

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(input_dim, latent_dim))
            else:
                self.hidden_layers.append(torch.nn.Linear(latent_dim, latent_dim))
            self.hidden_layers.append(torch.nn.BatchNorm1d(latent_dim))
            self.hidden_layers.append(torch.nn.ReLU(True))

        self.output_layer = torch.nn.Linear(latent_dim, output_dim)

    def init_weights(self):
        """Initialize weights of the MLP layers."""
        for module in self.hidden_layers:
            if isinstance(module, torch.nn.Linear):
                # Kaiming is better
                if isinstance(self.hidden_init, torch.nn.init.kaiming_normal_):
                    self.hidden_init(module.weight, nonlinearity="relu")
                elif isinstance(self.hidden_init, torch.nn.init.normal_):
                    self.hidden_init(module.weight, mean=0.0, std=0.01)
                else:
                    raise AssertionError("Unsupported initialization type")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
        if isinstance(self.output_init, torch.nn.init.kaiming_normal_):
            self.output_init(self.output_layer.weight, nonlinearity="relu")
        elif isinstance(self.output_init, torch.nn.init.normal_):
            self.output_init(self.output_layer.weight, mean=0.0, std=0.01)
        else:
            raise AssertionError("Unsupported initialization type")
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


@dataclass
class Config:
    # Basic
    disable_viewer: bool = False
    ckpt: Optional[List[str]] = None
    render_traj_path: str = "ellipse"
    tb_every: int = 100
    tb_save_image: bool = False
    lpips_net: Literal["vgg", "alex"] = "vgg"
    port: int = 8080
    batch_size: int = 1
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    data_dir: str = "examples/data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results"
    test_every: int = 8
    patch_size: Optional[int] = None
    normalize_world_space: bool = True
    near_plane: float = 0.01
    far_plane: float = 1e10

    # Model
    feat_dim: int = 256
    input_dim: int = 256
    num_layers: int = 6
    n_feat_offsets: int = 10
    n_gauss: int = 10_000

    # Loss
    ssim_lambda: float = 0.2
    scale_reg: float = 0.01
    opacity_reg: float = 0.01
    offset_reg: float = 0.1
    depth_loss: bool = False
    depth_lambda: float = 1e-2
    chamfer_lambda: float = 1e-2
    photo_loss_lambda: float = 0
    lr: float = 5e-4


def create_splats_with_optimizers(
    cfg: Config,
    sparse_grad: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    world_size: int = 1,
) -> tuple[
    dict[str, ModuleDict | ParameterDict], dict[str, dict[str, SparseAdam | Adam]]
]:
    # Define gauss_params
    gauss_params = torch.nn.ParameterDict(
        {
            "anchors": torch.nn.Parameter(
                torch.randn((cfg.n_gauss, cfg.input_dim), device=device)
            )
        }
    ).to(device)

    geo_mlp: torch.nn.Sequential = MLP(
        input_dim=cfg.input_dim,
        latent_dim=512,
        output_dim=3 * cfg.n_feat_offsets,
        num_layers=4,
    ).cuda()
    pos_enc = PositionalEncoding(num_bases=10, include_input=True).cuda()

    app_mlp: torch.nn.Sequential = MLP(
        input_dim=pos_enc.output_dim,
        latent_dim=512,
        output_dim=11,
        num_layers=4,
        output_init=torch.nn.init.normal_,
    ).cuda()

    # Initialize decoders (MLPs)
    decoders = torch.nn.ModuleDict(
        {
            "geo_mlp": geo_mlp,
            "pos_enc": pos_enc,
            "app_mlp": app_mlp,
        }
    ).to(device)

    # Scale learning rates based on batch size (BS)
    BS = batch_size * world_size

    # Create optimizers for gauss_params
    gauss_optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": param, "lr": cfg.lr * math.sqrt(BS)}],
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
                    "lr": cfg.lr * math.sqrt(BS),
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
        self.sfm_points = torch.from_numpy(self.parser.points).float().to(self.device)
        self.scene_scale = self.parser.scene_scale * 1.1
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            cfg=cfg,
            batch_size=cfg.batch_size,
            device=self.device,
            world_size=world_size,
        )

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

        offsets = self.splats["decoders"]["geo_mlp"](
            self.splats["gauss_params"]["anchors"]
        )
        offsets = offsets.view(-1, self.cfg.n_feat_offsets, 3).view(-1, 3)
        enc = self.splats["decoders"]["pos_enc"](offsets)

        apps = self.splats["decoders"]["app_mlp"](enc)
        apps = apps.view(-1, self.cfg.n_feat_offsets, 11)
        vis_colors, vis_opacity, quats, scales = apps.split([3, 1, 4, 3], dim=-1)
        vis_colors = vis_colors.view(-1, 3)
        vis_opacity = vis_opacity.view(-1, 1)
        quats = quats.view(-1, 4)
        scales = scales.view(-1, 3)

        info = {
            "means": offsets * self.scene_scale * 1.0,
            "colors": vis_colors.sigmoid(),
            "opacities": vis_opacity.sigmoid()[:, 0],
            "scales": F.softplus(scales) * self.scene_scale * 1e-2,
            "quats": quats / quats.norm(dim=-1, keepdim=True),
        }

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
                gamma=0.01 ** (1.0 / max_steps),
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["decoders_optimizer"]["geo_mlp"],
                gamma=0.01 ** (1.0 / max_steps),
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["decoders_optimizer"]["app_mlp"],
                gamma=0.01 ** (1.0 / max_steps),
            ),
        ]

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
        pbar = tqdm.tqdm(range(init_step, max_steps))

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

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            loss *= cfg.photo_loss_lambda
            desc = f"loss={loss.item():.3f}| "

            # lpipsloss = self.lpips(
            #     colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
            # )
            # loss += lpipsloss * 0.1
            # desc += f"lpips loss={lpipsloss.item():.3f}| "
            # if cfg.scale_reg > 0:
            #     scale_loss = info["scales"].mean() * cfg.scale_reg
            #     loss += scale_loss
            #     desc += f"scale loss={scale_loss.item():.6f}| "
            # if cfg.opacity_reg > 0:
            #     opacity_loss = (1 - info["opacities"]).mean() * cfg.opacity_reg
            #     # opacity_loss = info["opacities"].mean() * cfg.opacity_reg
            #     loss += opacity_loss
            #     desc += f"opacity loss={opacity_loss.item():.6f}| "
            # if cfg.offset_reg > 0:
            #     offset_loss = info["offsets"].norm(dim=-1).mean() * cfg.offset_reg
            #     loss += offset_loss
            #     desc += f"offset loss={offset_loss.item():.6f}| "

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

            asym_chamfer_loss = asym_chamfer(
                info["means"], self.sfm_points
            ) + asym_chamfer(self.sfm_points, info["means"])
            loss += asym_chamfer_loss * cfg.chamfer_lambda
            desc += f"chamfer loss={asym_chamfer_loss.item():.6f}| "

            loss.backward()

            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(info["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                self.writer.flush()

            # optimize
            for optimizer in self.optimizers["gauss_optimizer"].values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # optimize
            for optimizer in self.optimizers["decoders_optimizer"].values():
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
            colors, _, info = self.rasterize_splats(
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

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(info["means"]),
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
    cli(main, cfg, verbose=True)
