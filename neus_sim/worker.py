from collections import OrderedDict
import os
import shutil
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .loss import reconst_loss, eikonal_loss, tvl1_loss, entropy_loss
from .neus import SDFNet, DeviationNet, BRDFNet, NeuSRenderer
from .train_utils import (
    LinearWarmupCosineAnnealingLR,
    Logger, AverageMeter, fix_random_seed, time_str
)


@torch.jit.script
def convolve(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv1d(
        x.unsqueeze(0), kernel,
        padding=x.shape[-1] // 2 - 1, groups=kernel.shape[0]
    )[0]


@torch.jit.script
def run_sensor_model(
    hists: torch.Tensor,
    pulse_kernel: torch.Tensor,
    jitter_kernel: torch.Tensor,
    power: float,
    noise: float,
) -> torch.Tensor:
    """ Converts impulse response into sensor response. """

    hists = power * convolve(hists, pulse_kernel) + noise
    r_sum = torch.cumsum(
        torch.cat((hists.new_zeros(hists.shape[0], 1), hists[:, :-1]), dim=1),
        dim=1
    )
    hists = (1 - torch.exp(-hists)) * torch.exp(-r_sum)
    hists = convolve(hists, jitter_kernel)
    return hists


class Trainer:

    def __init__(self, opt):

        self.opt = opt

        rng = fix_random_seed(opt.get("seed", 2023))

        # build model
        self.renderer = NeuSRenderer(
            sdf_net=SDFNet(**opt["sdf"]),
            deviation_net=DeviationNet(**opt["deviation"]),
            brdf_net=BRDFNet(**opt["brdf"]),
        ).cuda()

        # prepare data
        data_path = os.path.join(opt["dataset"], "sensor_response.npz")
        print(data_path)
        assert os.path.exists(data_path)
        data = np.load(data_path)
        self.batch_size = opt["batch_size"]
        
        self.fov = float(data["fov"])               # field of view
        self.bin_size = float(data["bin_size"]) / 2 # depth resolution
        self.num_bins = int(data["num_bins"])       # histogram size
        self.power = float(data["power"])           # laser power
        self.noise = float(data["noise"])           # noise floor

        Rts = data["Rts"].astype(np.float32)
        hists = (data["hists_jitter"] / data["num_cycles"]).astype(np.float32)

        self.num_views = len(Rts)                   # number of views
        self.Rts = torch.from_numpy(Rts).cuda()     # camera-to-world transformation
        self.hists = torch.from_numpy(hists).cuda() # ground-truth histograms

        pulse_kernel = data["pulse_kernel"].astype(np.float32)
        pulse_kernel = torch.from_numpy(pulse_kernel).cuda()
        pulse_kernel = pulse_kernel.repeat(self.batch_size, 1).unsqueeze(1)
        self.pulse_kernel = torch.flip(pulse_kernel, dims=(-1, ))
        
        jitter_kernel = data["jitter_kernel"].astype(np.float32)
        jitter_kernel = torch.from_numpy(jitter_kernel).cuda()
        jitter_kernel = jitter_kernel.repeat(self.batch_size, 1).unsqueeze(1)
        self.jitter_kernel = torch.flip(jitter_kernel, dims=(-1, ))

        # rendering options
        self.num_ray_samples = opt["render"]["num_ray_samples"]
        self.num_step_samples = opt["render"]["num_step_samples"]
        self.radius = opt["render"]["radius"]

        # build training utilities
        self.num_itrs = opt["itrs"] + opt["warmup_itrs"]
        self.num_cos_anneal_itrs = opt.get("cos_anneal_itrs", 0)
        self.itr = 0
        self.optimizer = torch.optim.Adam(
            params=self.renderer.parameters(),
            lr=opt["lr"],
            betas=opt.get("betas", (0.9, 0.999)),
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self.optimizer,
            warmup_epochs=opt["warmup_itrs"],
            max_epochs=self.num_itrs,
            eta_min=opt.get("min_lr", 0),
        )
        self.clip_grad_norm = opt.get("clip_grad_norm")

        # loss weights
        self.reconst_mode = opt.get("reconst_mode", "mse")
        self.reconst_log = opt.get("reconst_log", False)
        self.eikonal_weight = opt.get("eikonal_weight", 0)
        self.tvl1_weight = opt.get("tvl1_weight", 0)
        self.entropy_weight = opt.get("entropy_weight", 0)

        # build logging utilities
        self.log_interval = opt.get("log_interval", 100)
        self.plot_interval = opt.get("plot_interval", 1000)
        self.checkpoint_interval = opt.get("checkpoint_interval", 10000)
        
        self.logger = Logger(os.path.join(opt["_root"], "log.txt"))
        self.tb_writer = SummaryWriter(os.path.join(opt["_root"], "tensorboard"))
        self.inv_s_tracker, self.rho_tracker = AverageMeter(), AverageMeter()
        self.loss_meters = OrderedDict()
        self.timer = AverageMeter()

        # load model weights and training states
        if opt["_resume"]:
            self.load()

    def cosine_annealing(self):
        if not self.num_cos_anneal_itrs:
            return 1.0
        return min(1.0, self.itr / self.num_cos_anneal_itrs)

    def forward_backward(self, Rt, gt_hists):
        cos_anneal_ratio = self.cosine_annealing()

        # impulse response
        pred_dict = self.renderer(
            Rt=Rt,
            fov=self.fov,
            bin_size=self.bin_size,
            num_bins=self.num_bins,
            num_ray_samples=self.num_ray_samples,
            num_step_samples=self.num_step_samples,
            cos_anneal_ratio=cos_anneal_ratio,
            radius=self.radius,
        )

        # sensor response
        pred_hists = run_sensor_model(
            hists=pred_dict["hists"],
            pulse_kernel=self.pulse_kernel,
            jitter_kernel=self.jitter_kernel,
            power=self.power,
            noise=self.noise,
        )
        pred_dict["convolved_hists"] = pred_hists.detach()
        
        # compute loss
        loss_dict = OrderedDict()
        loss = reconst_loss(
            pred_hists, gt_hists, mode=self.reconst_mode, log=self.reconst_log
        )
        loss_dict["mse"] = loss.clone().detach()
        if self.eikonal_weight > 0:
            eikonal = eikonal_loss(pred_dict["normal"], pred_dict["inside_sphere"])
            loss += self.eikonal_weight * eikonal
            loss_dict["eikonal"] = eikonal.clone().detach()
        if self.tvl1_weight > 0:
            tvl1 = tvl1_loss(pred_dict["alpha"])
            loss += self.tvl1_weight * tvl1
            loss_dict["tvl1"] = tvl1.clone().detach()
        if self.entropy_weight > 0:
            entropy = entropy_loss(pred_dict["opacity"])
            loss += cos_anneal_ratio * self.entropy_weight * entropy
            loss_dict["entropy"] = entropy.clone().detach()

        loss.backward()
        loss_dict["total"] = loss.clone().detach()
        for k, v in pred_dict.items():
            pred_dict[k] = v.detach()

        return pred_dict, loss_dict

    def run(self):
        print("Training started.")
        while self.itr < self.num_itrs:
            # run one optimization step
            start_time = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            data_idx = random.sample(range(self.num_views), self.batch_size)
            Rt, gt_hists = self.Rts[data_idx], self.hists[data_idx]
            pred_dict, loss_dict = self.forward_backward(Rt, gt_hists)
            
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.renderer.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            self.itr += 1
            
            # log
            self.inv_s_tracker.update(pred_dict["inv_s"].detach())
            self.rho_tracker.update(pred_dict["rho"].detach())
            for k, v in loss_dict.items():
                if k not in self.loss_meters:
                    self.loss_meters[k] = AverageMeter()
                self.loss_meters[k].update(v.detach())
            self.timer.update(time.time() - start_time)
            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
            if self.itr == 1 or self.itr % self.plot_interval == 0:
                self.plot(pred_dict, gt_hists)
            if self.itr % self.checkpoint_interval == 0:
                self.checkpoint()
        
        self.checkpoint()
        print("Training completed.")

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.4f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        self.tb_writer.add_scalar("inv_s", self.inv_s_tracker.item(), self.itr)
        self.tb_writer.add_scalar("rho", self.rho_tracker.item(), self.itr)
        self.tb_writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()

    def plot(self, pred_dict, gt_hists):
        nv = min(pred_dict["hists"].shape[0], 4)
        pred_hists = pred_dict["convolved_hists"][:nv].cpu().numpy()
        gt_hists = gt_hists[:nv].cpu().numpy()

        nr = pred_dict["bin_alpha"].shape[1]
        r_idx = np.random.choice(np.arange(nr), min(nr, 8), replace=False)
        bin_alpha = pred_dict["bin_alpha"][:nv, r_idx].cpu().numpy()
        bin_weights = pred_dict["bin_weights"][:nv, r_idx].cpu().numpy()
        opacity = pred_dict["opacity"].flatten().cpu().numpy()
        
        fig, axs = plt.subplots(nv, 4, sharex=True, sharey=False)
        x_tics = np.arange(gt_hists.shape[-1])
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, len(r_idx))]
        for i in range(nv):
            axs[i, 0].plot(x_tics, pred_hists[i], c="g")
            axs[i, 0].plot(x_tics, gt_hists[i], c="b")
            for j, color in enumerate(colors):
                axs[i, 1].plot(x_tics, bin_alpha[i, j], c=color)
                axs[i, 2].plot(x_tics, bin_weights[i, j], c=color)
        gs = axs[0, -1].get_gridspec()
        for ax in axs[:, -1]:
            ax.remove()
        ax = fig.add_subplot(gs[:, -1])
        ax.hist(opacity, bins=np.linspace(0, 1, 20), orientation="horizontal")
        fig.tight_layout()

        t = len(str(self.num_itrs))
        self.tb_writer.add_figure(
            tag=f"{self.itr:0{t}d}", figure=fig, global_step=self.itr
        )

    def load(self):
        model_path = os.path.join(self.opt["_root"], "models", "last.pth")
        state_path = os.path.join(self.opt["_root"], "states", "last.pth")
        model_ckpt = torch.load(model_path, map_location="cpu")
        state_ckpt = torch.load(state_path, map_location="cpu")
        self.renderer.load_state_dict(model_ckpt)
        self.optimizer.load_state_dict(state_ckpt["optimizer"])
        self.scheduler.load_state_dict(state_ckpt["scheduler"])
        self.itr = state_ckpt["itr"]
        t = len(str(self.num_itrs))
        print(f"Loaded checkpoint [itr {self.itr:0{t}d}]...")

    def checkpoint(self):
        t = len(str(self.num_itrs))
        print(f"Checkpointing at [itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt["_root"], "models")
        state_dir = os.path.join(self.opt["_root"], "states")
        model_ckpt = self.renderer.state_dict()
        state_ckpt = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "itr": self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, "last.pth"))
        torch.save(state_ckpt, os.path.join(state_dir, "last.pth"))
        shutil.copyfile(
            os.path.join(model_dir, "last.pth"),
            os.path.join(model_dir, f"{self.itr:0{t}d}.pth")
        )


class MeshExtractor:

    def __init__(self, opt):

        self.opt = opt

        # build model
        renderer = NeuSRenderer(
            sdf_net=SDFNet(**opt["sdf"]),
            deviation_net=DeviationNet(**opt["deviation"]),
            brdf_net=BRDFNet(**opt["brdf"]),
        )

        # load checkpoint
        model_ckpt = torch.load(opt["ckpt_path"], map_location="cpu")
        renderer.load_state_dict(model_ckpt)
        self.sdf_net = renderer.sdf_net.cuda()
        self.sdf_net.eval()

    def is_inside_hemisphere(self, pts, radius):
        """ Determine whether points are within the upper hemisphere. """

        return torch.logical_and(
            torch.linalg.vector_norm(pts, dim=-1) < radius, pts[..., -1] > 0
        )

    def run(self):
        print("Extracting mesh from SDF...")

        radius = self.opt["render"]["radius"]
        resolution = self.opt["grid_resolution"]
        assert resolution % 64 == 0

        # prepare sampling grid
        X = torch.linspace(-radius, radius, resolution, device="cuda").split(64)
        Y = torch.linspace(-radius, radius, resolution, device="cuda").split(64)
        Z = torch.linspace(0, radius, resolution // 2, device="cuda").split(64)

        # divide reconstruction volume into 64x64x64 cubes for SDF evaluation
        ## this is to avoid GPU memory overflow
        vol = torch.zeros((resolution, resolution, resolution // 2), device="cuda")
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                        pts = torch.cat(
                            (
                                xx.reshape(-1, 1),
                                yy.reshape(-1, 1),
                                zz.reshape(-1, 1)
                            ), dim=-1
                        )
                        val = -self.sdf_net(pts).squeeze(-1)
                        # points outside hemisphere are non considered
                        val[~self.is_inside_hemisphere(pts, radius)] = -1000
                        vol[
                            xi * 64 : xi * 64 + len(xs),
                            yi * 64 : yi * 64 + len(ys),
                            zi * 64 : zi * 64 + len(zs)
                        ] = val.reshape(len(xs), len(ys), len(zs))
        vol = vol.cpu().numpy()
        vol = np.concatenate(
            (vol, -1000 * np.ones((resolution, resolution, resolution // 2))),
            axis=-1
        )

        # run marching cubes to extract mesh
        try:
            import mcubes
        except:
            raise ImportError("mcubes not installed, run 'pip install PyMCubes'")
        vertices, faces = mcubes.marching_cubes(vol, 0)
        vertices = vertices / (resolution - 1.0) * radius * 2 - radius  # center mesh

        # save mesh
        np.savez(
            os.path.join(self.opt["_root"], "mesh.npz"),
            vertices=vertices, faces=faces
        )
        print("Mesh extraction completed.")

        if self.opt["viz"]:
            try:
                import trimesh
            except:
                raise ImportError("trimesh not installed, run 'pip install trimesh'")
            mesh = trimesh.Trimesh(vertices, faces)
            mesh.vertices -= mesh.center_mass
            mesh.export(os.path.join(self.opt['_root'], 'viz_thresh{}.stl'.format(0)))
            print('mesh is: ', mesh)
            #mesh.show()