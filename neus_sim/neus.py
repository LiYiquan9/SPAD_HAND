import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial


@torch.jit.script
def position_encoding(x, freqs):
    """ Map xyz-coordinates to high-dimensional features. """

    h = (x.unsqueeze(-1) * freqs).flatten(-2)
    embd = torch.cat((x, torch.sin(h), torch.cos(h)), dim=-1)
    return embd


class SDFNet(nn.Module):
    """
    Signed distance function of an object.
    Input is xyz-coordinates of a 3D point in world coordinate.
    """

    def __init__(
        self,
        hid_dim=256,        # hidden feature size
        num_freqs=6,        # number of frequencies for position encoding
        num_layers=8,       # number of layers in the network
        skips=(4, ),        # layers with skip connection
        weight_norm=True,   # whether to apply weight normalization
        geometric_init=True,# whether to apply geometric initialization
        r_init=0.5,         # radius of sphere at initialization
    ):
        super().__init__()

        # position encoding
        ## assume input size is 3 (xyz coordinates)
        freqs = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = (num_freqs * 2 + 1) * 3

        self.layers = nn.ModuleList()
        dims = [in_dim] + [hid_dim] * num_layers + [1]
        self.num_layers = len(dims)
        
        for l in range(self.num_layers - 1):
            out_dim = dims[l + 1] - dims[0] if l + 1 in skips else dims[l + 1]
            layer = nn.Linear(dims[l], out_dim)
            self.layers.append(layer)
            
            # geometric initialization (https://arxiv.org/abs/1911.10414)
            ## SDF initialized to approximate a sphere with radius r
            ## c.f. https://arxiv.org/abs/2003.09852
            if geometric_init:
                if l == self.num_layers - 2:    # output layer
                    mean = np.sqrt(np.pi / dims[l])
                    nn.init.normal_(layer.weight, mean=mean, std=1e-4)
                    nn.init.constant_(layer.bias, -r_init)
                elif l == 0:    # input layer
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.constant_(layer.weight[:, 3:], 0.0)
                    std = np.sqrt(2 / out_dim)
                    nn.init.normal_(layer.weight[:, :3], mean=0.0, std=std)
                elif l in skips:    # layer with skip input
                    nn.init.constant_(layer.bias, 0.0)
                    std = np.sqrt(2 / out_dim)
                    nn.init.normal_(layer.weight, mean=0.0, std=std)
                    nn.init.constant_(layer.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    std = np.sqrt(2 / out_dim)
                    nn.init.normal_(layer.weight, mean=0.0, std=std)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

        self.activation = nn.Softplus(beta=100) # smooth ReLU
        self.skips = skips

    def forward(self, x):
        x = position_encoding(x, self.freqs)
        inputs = x
        for l, layer in enumerate(self.layers):
            if l in self.skips:
                x = torch.cat((x, inputs), dim=-1) / np.sqrt(2)
            x = layer(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        x = x.squeeze(-1)
        return x


class DeviationNet(nn.Module):
    """ Standard deviation (1 / s) of logistic density function. """

    def __init__(self, init_val=0.3):
        super().__init__()

        self.val = nn.Parameter(torch.tensor(init_val))

    @property
    def inv_s(self):
        return torch.clamp(torch.exp(self.val * 10.0), min=1e-4, max=1e4)

    def forward(self, x, inv_s=None):
        return (inv_s or self.inv_s) * x


class BRDFNet(nn.Module):
    """ Lambertian BRDF with homogeneous surface albedo. """

    def __init__(self, init_val=0.0):
        super().__init__()

        self.val = nn.Parameter(torch.tensor(init_val))

    @property
    def rho(self):
        return torch.sigmoid(self.val)

    def forward(self, x, rho=None):
        return (rho or self.rho) / np.pi * x


class NeuSRenderer(nn.Module):
    """ Volume renderer for transient histograms. """

    def __init__(self, sdf_net, deviation_net, brdf_net):
        super().__init__()

        self.sdf_net = sdf_net
        self.deviation_net = deviation_net
        self.brdf_net = brdf_net

    @torch.no_grad()
    def sample_rays(
        self,
        Rt,                     # (v, 3, 4), camera-to-world transformations
        fov,                    # conic field of view of histograms (unit: degrees)
        num_samples,            # number of rays drawn from uniform distribution
        pdf=None,               # (v, r), probability density function (PDF)
        num_pdf_samples=None,   # number of samples drawn from the supplied PDF
        device="cuda"
    ):
        """ Sample rays for each view. """

        nv = Rt.shape[0]  # number of views
        u_size, v_size = num_samples
        num_samples = u_size * v_size

        # uniform sampling
        u_tics = torch.linspace(0, u_size, u_size + 1, device=device)[:-1]
        v_tics = torch.linspace(0, v_size, v_size + 1, device=device)[:-1]
        u, v = torch.meshgrid(u_tics, v_tics, indexing="ij")
        u, v = u.flatten(), v.flatten()
        ru = (u + torch.rand(nv, num_samples, device=device)) / u_size
        rv = (v + torch.rand(nv, num_samples, device=device)) / v_size
        uv_idx = torch.arange(num_samples, device=device).repeat(nv, 1) # (v, r)

        # weighted sampling
        if pdf is not None:
            assert pdf.shape[-1] == num_samples
            assert num_pdf_samples is not None
            cnts = (
                Multinomial(num_pdf_samples, probs=pdf + 1e-6)
                .sample().int().flatten()
            )
            uv_idx_pdf = (
                torch.arange(num_samples, device=device)
                .repeat(nv).repeat_interleave(cnts)
            )
            u = u[uv_idx_pdf].reshape(nv, num_pdf_samples)
            v = v[uv_idx_pdf].reshape(nv, num_pdf_samples)
            ru_pdf = (u + torch.rand_like(u)) / u_size
            rv_pdf = (v + torch.rand_like(v)) / v_size
            ru = torch.cat((ru, ru_pdf), dim=-1)
            rv = torch.cat((rv, rv_pdf), dim=-1)
            w = 1 / ((cnts + 1) * num_samples)  # (v * r, )
            w_pdf = w.repeat_interleave(cnts).reshape(nv, num_pdf_samples)
            w = w.reshape(nv, num_samples)
            w = torch.cat((w, w_pdf), dim=-1)
            uv_idx_pdf = uv_idx_pdf.reshape(nv, num_pdf_samples)
            uv_idx = torch.cat((uv_idx, uv_idx_pdf), dim=-1)
        else:
            w = torch.ones_like(ru) / num_samples   # (v, r)

        # map uv-coordinates to ray directions
        max_phi = 2 * np.pi
        min_cos_theta = np.cos(fov * np.pi / 360)
        solid_angle = 2 * np.pi * (1 - min_cos_theta)        
        phi = ru * max_phi
        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        cos_theta = rv * (1 - min_cos_theta) + min_cos_theta
        sin_theta = (1 - cos_theta ** 2) ** 0.5
        d = torch.stack(
            (sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), dim=-1
        )   # (v, r, 3)

        # transform to world coordinates
        o = Rt[..., -1] # (v, 3)
        d = torch.einsum("vnj,vij->vni", d, Rt[..., :3])    # (v, r, 3)
        w = w * cos_theta * solid_angle # cosine attenuation at sensor
        return o, d, w, uv_idx

    @torch.no_grad()
    def sample_steps(
        self,
        v,                      # number of views
        r,                      # number of rays per view
        bin_size,               # depth range covered by a bin (unit: m)
        num_bins,               # number of bins in a histogram
        num_samples,            # number of samples drawn from uniform distribution
        pdf=None,               # (v, r, b), probability density function (PDF)
        num_pdf_samples=None,   # number of samples drawn from the supplied PDF
        device="cuda"
    ):
        """ Sample steps along each ray. """

        edges = bin_size * torch.linspace(
            0, num_bins, num_bins + 1, device=device
        ).repeat(v, r, 1)  # (v, r, b + 1)
        left_edges = edges[..., :-1].unsqueeze(-1)  # (v, r, b, 1)

        # uniform sampling
        assert num_samples % num_bins == 0
        etas = torch.rand(
            v, r, num_bins, num_samples // num_bins - 1, device=device
        ) * bin_size
        steps = torch.cat((left_edges, left_edges + etas), dim=-1)
        steps = torch.cat((steps.flatten(-2), edges[..., -1:]), dim=-1)
        
        # weighted sampling
        if pdf is not None:
            assert pdf.shape[0] == v and pdf.shape[1] == r
            assert num_pdf_samples is not None
            bin_idx = torch.multinomial(
                pdf.flatten(0, 1) + 1e-6, num_pdf_samples, replacement=True
            ).reshape(v, r, num_pdf_samples)
            left_edges = torch.gather(edges, dim=-1, index=bin_idx)
            etas = torch.rand_like(left_edges) * bin_size
            steps = torch.cat((steps, left_edges + etas), dim=-1)

        # sort steps in ascending order
        steps = torch.sort(steps, dim=-1)[0]
        return steps

    def is_inside_hemisphere(self, pts, radius):
        """ Determine whether points are within the upper hemisphere. """

        return torch.logical_and(
            torch.linalg.vector_norm(pts, dim=-1) < radius, pts[..., -1] > 0
        )

    @torch.no_grad()
    def run_coarse(
        self,
        Rt,                 # (v, 3, 4), camera-to-world transformations
        fov,                # conic field of view of histograms (unit: degrees)
        bin_size,           # depth range covered by a bin (unit: m)
        num_bins,           # number of bins in a histogram
        num_ray_samples,    # number of rays per view
        num_step_samples,   # number of steps per ray
        radius=1.0,         # sphere radius (unit: m)
    ):
        """ Run coarse sampling to obtain PDFs for fine sampling. """

        device = Rt.device

        # sample rays and steps along a ray
        if not isinstance(num_ray_samples, (list, tuple)):
            num_ray_samples = (num_ray_samples, num_ray_samples)
        assert len(num_ray_samples) == 2
        o, d, _, _ = self.sample_rays(Rt, fov, num_ray_samples, device=device)
        nv, nr = d.shape[:2]    # number of views and number of rays
        steps = self.sample_steps(
            nv, nr, bin_size, num_bins, num_step_samples, device=device
        ) # (v, r, p)
        deltas = steps[..., 1:] - steps[..., :-1]       # segment lengths

        # evaluate SDF at sampled points
        pts = o[:, None, None, :] + steps[..., None] * d[..., None, :]  # (v, r, p, 3)
        sdf = self.sdf_net(pts) # (v, r, p)

        # evaluate opacities
        mid_pts = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        mid_sdf = 0.5 * (sdf[..., 1:] + sdf[..., :-1])
        cos = (sdf[..., 1:] - sdf[..., :-1]) / (deltas + 1e-6)  # SDF linearization
        prev_cos = torch.cat(
            (cos.new_zeros(*cos.shape[:-1], 1), cos[..., :-1]), dim=-1
        )
        cos = torch.clamp(torch.minimum(prev_cos, cos), min=-1e3, max=0)    # non-positive
        dsdf = 0.5 * cos * deltas
        prev_T = torch.sigmoid(64 * (mid_sdf - dsdf))
        next_T = torch.sigmoid(64 * (mid_sdf + dsdf))
        alpha = 1 - next_T / (prev_T + 1e-6)    # (v, r, p)
        alpha *= self.is_inside_hemisphere(mid_pts, radius)

        # obtain alpha-blending weights
        T = torch.cumprod(  # round-trip transmittance
            torch.cat(
                (
                    alpha.new_ones(*alpha.shape[:-1], 1),
                    (1 - alpha[..., :-1]) ** 2 + 1e-6
                ), dim=-1
            ), dim=-1
        )
        weights = alpha * T # (v, r, p)

        # derive PDFs from weights
        weights = weights.reshape(nv, nr, num_bins, -1)
        step_pdf = torch.sum(weights, dim=-1)   # (v, r, b)
        ray_pdf = torch.sum(step_pdf, dim=-1)   # (v, r)
        return ray_pdf, step_pdf

    def run_fine(
        self,
        Rt,                 # (v, 3, 4), camera-to-world transformations
        fov,                # conic field of view of histograms (unit: degrees)
        bin_size,           # depth range covered by a bin (unit: m)
        num_bins,           # number of bins in a histogram
        ray_pdf,            # (v, r), PDF for adaptive ray sampling
        step_pdf,           # (v, r, b), PDF for adaptive step sampling
        num_ray_samples,    # number of rays per view
        num_step_samples,   # number of steps per ray
        cos_anneal_ratio=0, # cosine annealing ratio
        radius=1,           # sphere radius (unit: m)
    ):
        """ Run fine sampling to obtain rendered histograms. """

        device = Rt.device

        # sample rays and steps along a ray
        o, d, w, uv_idx = self.sample_rays(
            Rt, fov, num_ray_samples[0], ray_pdf, num_ray_samples[1], device=device
        )
        nv, nr = d.shape[:2]
        step_pdf = torch.gather(    # collect step PDFs for sampled rays
            step_pdf, dim=1,
            index=torch.tile(uv_idx.unsqueeze(-1), (1, 1, step_pdf.shape[-1]))
        )   # (v, r, b)
        steps = self.sample_steps(
            nv, nr, bin_size, num_bins, num_step_samples[0],
            step_pdf, num_step_samples[1], device=device
        )
        deltas = steps[..., 1:] - steps[..., :-1]           # segment lengths
        steps = 0.5 * (steps[..., 1:] + steps[..., :-1])    # mid-point rule

        # evaluate SDF, normal and cosine attenuation at sampled points
        pts = o[:, None, None, :] + steps[..., None] * d[..., None, :]  # (v, r, p, 3)
        pts_shape = pts.shape
        inside_sphere = self.is_inside_hemisphere(pts, radius)

        pts = pts[inside_sphere]    # only run SDF network on valid points
        sdf = self.sdf_net(pts.requires_grad_(True))    # (v, r, p)
        normal = torch.autograd.grad(
            sdf, pts, grad_outputs=torch.ones_like(sdf),
            create_graph=True, only_inputs=True
        )[0]  # (v, r, p, 3)
        sdf = sdf.new_zeros(*pts_shape[:-1]).masked_scatter_(inside_sphere, sdf)
        normal = (
            normal.new_zeros(*pts_shape)
            .masked_scatter_(inside_sphere.unsqueeze(-1), normal)
        )

        ## ray directions (d) are unit vectors
        ## normals will approach unit length for legitimate SDF
        cos = torch.einsum("vrj,vrpj->vrp", d, normal)  # (v, r, p)
        annealed_cos = -(   # prevent cosines all going to zero at initialization
            F.relu(-cos * 0.5 + 0.5, inplace=True) * (1 - cos_anneal_ratio)
            + F.relu(-cos, inplace=True) * cos_anneal_ratio
        )   # non-positive

        # evaluate opacities
        dsdf = 0.5 * annealed_cos * deltas
        prev_T = torch.sigmoid(self.deviation_net(sdf - dsdf))
        next_T = torch.sigmoid(self.deviation_net(sdf + dsdf))
        alpha = 1 - next_T / (prev_T + 1e-6)    # (v, r, p)
        alpha = alpha * inside_sphere

        # obtain alpha-blending weights
        T = torch.cumprod(
            torch.cat(  # transmittance
                (alpha.new_ones(*alpha.shape[:-1], 1), (1 - alpha[..., :-1]) ** 2),
                dim=-1
            ), dim=-1
        )   # (v, r, p)
        weights = alpha * T # (v, r, p)

        # evaluate shading (quadratic fall-off)
        radiance = weights * self.brdf_net(-annealed_cos / (steps ** 2 + 1e-6)) # (v, r, p)

        # collect histogram values
        steps = steps.unsqueeze(-1) # (v, r, p, 1)
        edges = bin_size * torch.linspace(0, num_bins, num_bins + 1, device=device)
        idx_mat = torch.logical_and(steps > edges[:-1], steps < edges[1:]).float()   # (v, r, p, b)
        bin_alpha = torch.einsum("vrpb,vrp->vrb", idx_mat, alpha)       # (v, r, b)
        bin_weights = torch.einsum("vrpb,vrp->vrb", idx_mat, weights)   # (v, r, b)
        opacity = torch.sum(bin_weights, dim=-1)                        # (v, r)
        radiance = torch.einsum("vrpb,vrp->vrb", idx_mat, radiance)     # (v, r, b)
        hists = torch.sum(radiance * w.unsqueeze(-1), dim=-2)           # (v, b)
        return {
            "hists": hists,                 # (v, b)

            "sdf": sdf,                     # (v, r, p)
            "inside_sphere": inside_sphere, # (v, r, p)
            "normal": normal,               # (v, r, p, 3)
            
            "alpha": alpha,                 # (v, r, p)
            "T": T,                         # (v, r, p)
            "weights": weights,             # (v, r, p)

            "bin_alpha": bin_alpha,         # (v, r, b)
            "bin_weights": bin_weights,     # (v, r, b)
            "opacity": opacity,             # (v, r)

            "inv_s": self.deviation_net.inv_s,
            "rho": self.brdf_net.rho,
        }

    def forward(
        self,
        Rt,
        fov,
        bin_size,
        num_bins,
        num_ray_samples=[[64, 8], 64],
        num_step_samples=[256, 32],
        cos_anneal_ratio=0,
        radius=1,
    ):
        nv = Rt.shape[0]    # number of views

        # coarse sampling
        ray_pdf, step_pdf = self.run_coarse(
            Rt, fov, bin_size, num_bins,
            num_ray_samples[0], num_step_samples[0], radius
        )

        # fine sampling
        pred_dict = self.run_fine(
            Rt, fov, bin_size, num_bins,
            ray_pdf, step_pdf, num_ray_samples, num_step_samples,
            cos_anneal_ratio, radius
        )
        ray_pdf = ray_pdf.reshape(nv, *num_ray_samples[0])
        pred_dict["ray_pdf"] = ray_pdf   # (v, g, g)
        pred_dict["step_pdf"] = step_pdf # (v, r, b)
        return pred_dict