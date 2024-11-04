import os
import time
import tyro
from dataclasses import dataclass
import torch

import viser
import nerfview
from torch.utils.tensorboard import SummaryWriter

from datasets.colmap import Dataset, Parser
from utils import set_random_seed
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from collections import defaultdict
import json
import imageio
import numpy as np
import tqdm
import torch.nn.functional as F
from fused_ssim import fused_ssim
from pykeops.torch import generic_argkmin


@dataclass
class Config:
    ssim_lambda: float = 0.8
    lpips_lambda: float = 0
    depth_lambda: float = 1e-2
    chamfer_lambda: float = 1e-2
    tb_every: int = 100
    eval_every: int = 1000
    max_steps: int = 10000
    port: int = 8080
    data_dir: str = "examples/data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results"
    lpips_net: str = "alex"
    patch_size: Optional[int] = None
    depth_loss: bool = False
    normalize_world_space: bool = True
    test_every: int = 8
    batch_size: int = 1
    lr: float = 1e-3


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Dataset
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

        # Model & Optimizer
        # TODO

        # Losses & Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=cfg.lpips_net, normalize=True if cfg.lpips_net == "alex" else False
        ).to(self.device)

        # Viewer
        self.server = viser.ViserServer(port=cfg.port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    def rasterize_splats(
        self,
        c2ws: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor, Dict]:
        features = self.splats["decoders"]["feature_mlp"](
            self.splats["gauss_params"]["anchors"]
        )
        features = features.view(-1, self.cfg.n_feat_offsets, 14)
        offsets, vis_colors, vis_opacity, quats, scales = features.split(
            [3, 3, 1, 4, 3], dim=-1
        )
        offsets = offsets.view(-1, 3)
        vis_colors = vis_colors.view(-1, 3)
        vis_opacity = vis_opacity.view(-1, 1)
        quats = quats.view(-1, 4)
        scales = scales.view(-1, 3)

        info = {
            "means": means,
            "colors": colors.sigmoid(),
            "opacities": opacities.sigmoid(),
            "scales": F.softplus(scales) * self.scene_scale,
            "quats": quats / quats.norm(dim=-1, keepdim=True),
        }

        render_colors, render_alphas, raster_info = rasterization(
            means=info["means"],
            quats=info["quats"],
            scales=info["scales"],
            opacities=info["opacities"],
            colors=info["colors"],
            viewmats=torch.linalg.inv(c2ws),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            rasterize_mode="classic",
        )
        raster_info.update(info)
        return render_colors, render_alphas, raster_info

    def train(self):
        cfg = self.cfg
        device = self.device

        # TODO: scheduler

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        pbar = tqdm.tqdm(range(cfg.max_steps))

        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            Ks = data["K"].to(device)  # [1, 3, 3]
            c2ws = data["camtoworld"].to(device)  # [1, 4, 4]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            H, W = pixels.shape[1:3]

            if self.load_depths:
                # project sfm points to image plane to supervise depth
                points_gt = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            renders, alphas, info = self.rasterize_splats(
                c2ws=c2ws,
                Ks=Ks,
                width=W,
                height=H,
                render_mode="RGB+ED" if self.load_depths else "RGB",
            )
            colors, depths = renders[..., 0:3], renders[..., 3:4]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            depths = depths.permute(0, 3, 1, 2)  # [1, 1, H, W]
            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]

            l1_loss = F.l1_loss(colors, pixels)
            ssim_loss = (
                1.0 - fused_ssim(colors, pixels, padding="valid")
            ) * cfg.ssim_lambda
            lpips_loss = self.lpips(colors, pixels) * cfg.lpips_lambda

            # depth loss are calculated in disparity space
            points_gt = torch.stack(
                [
                    points_gt[:, :, 0] / (W - 1) * 2 - 1,
                    points_gt[:, :, 1] / (H - 1) * 2 - 1,
                ],
                dim=-1,
            )  # [1, M, 2]
            grid = points_gt[:, :, None, :]  # [1, M, 1, 2]
            depths = F.grid_sample(depths, grid, align_corners=True)  # [1, 1, M, 1]
            depths = depths[:, 0, :, 0]  # [1, M]
            disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            disp_gt = torch.where(
                depths_gt > 0.0, 1.0 / depths_gt, torch.zeros_like(depths_gt)
            )
            depth_loss = F.l1_loss(disp, disp_gt) * self.scene_scale * cfg.depth_lambda

            # supervise from sfm points
            knn1 = generic_argkmin(
                "SqDist(x, y)",
                "a = Vi(1)",
                "x = Vi(3)",
                "y = Vj(3)",
            )
            nn_indices = knn1(points_gt, info["means"])  # [N_sfm, 1]
            nn_points = info["means"][nn_indices]  # [N_sfm, 1, 3]
            chamfer_loss = (
                (nn_points - points_gt[:, None, :]).norm(dim=-1).mean()
            ) * cfg.chamfer_lambda

            loss = l1_loss + ssim_loss + lpips_loss + depth_loss + chamfer_loss
            desc = f"loss={loss:.6f}| l1={l1_loss:.6f}| ssim={ssim_loss:.6f}| lpips={lpips_loss:.6f}| depth={depth_loss:.6f}| chamfer={chamfer_loss:.6f}|"
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1", l1_loss.item(), step)
                self.writer.add_scalar("train/ssim", ssim_loss.item(), step)
                self.writer.add_scalar("train/lpips", lpips_loss.item(), step)
                self.writer.add_scalar("train/depth", depth_loss.item(), step)
                self.writer.add_scalar("train/chamfer", chamfer_loss.item(), step)
                self.writer.add_scalar("train/num_GS", info["means"].shape[0], step)
                self.writer.flush()

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # for scheduler in schedulers:
            #     scheduler.step()

            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

            self.viewer.lock.release()
            # Update the viewer state.
            self.viewer.update(step)

    def eval(self, step: int):
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            Ks = data["K"].to(device)  # [1, 3, 3]
            c2ws = data["camtoworld"].to(device)  # [1, 4, 4]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            H, W = pixels.shape[1:3]

            colors, _, info = self.rasterize_splats(
                c2ws=c2ws,
                Ks=Ks,
                width=W,
                height=H,
            )  # [1, H, W, 3]

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(f"{self.render_dir}/{step}_{i:04d}.png", canvas)

            pixels = pixels.permute(0, 2, 3, 1)  # [1, H, W, 3]
            colors = colors.permute(0, 2, 3, 1)  # [1, H, W, 3]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update({"num_GS": info["means"].shape[0]})
        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
            f"Number of GS: {stats['num_GS']}"
        )

        with open(f"{self.stats_dir}/{step:04d}.json", "w") as f:
            json.dump(stats, f)
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()


def main(cfg: Config):
    runner = Runner(cfg)
    runner.train()

    print("Viwer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)


"""
CUDA_VISIBLE_DEVICES=8 python simple_trainer_nocons.py --data_dir ~/dataset/360_v2/garden --data_factor 8 --max_steps 7000 \
  --strategy.refine_stop_iter -1 \
  --port 8096 \
  --result_dir results/debug_160t4
  
CUDA_VISIBLE_DEVICES=9 python simple_trainer_nocons_v2.py --data_dir ~/dataset/360_v2/garden --data_factor 8 --max_steps 7000 \
  --strategy.refine_stop_iter -1 \
  --port 8097 \
  --result_dir results/debug
"""
