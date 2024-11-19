import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import nvdiffrast.torch as dr
import torchvision
from pytorch3d.io import load_obj, save_obj

from manotorch.manolayer import ManoLayer
from meshes import Mesh, colors, hand_meshes

class SPADHistNV(nn.Module):
    def __init__(
        self,
        hand_batch_size=1,
        fov=33,
        radius=0.045,
        resolution=256,
        num_bins=64,
        # bin_size=0.014131013303995132,
        bin_size=0.0136,
        num_cameras=8,
        degree=70,
        fwhm=50,
        cube_mode=False,
        single_camera=False,
    ):
        super().__init__()

        self.resolution = resolution
        self.fov = fov

        self.num_cameras = num_cameras
        if single_camera:
            self.num_cameras = 1
        degree = degree / 180 * math.pi  # facing angle of the cameras
        self.radius = radius

        fov_rad = self.fov * math.pi / 180
        scale = math.tan(fov_rad / 2)
        self.width = scale * 2

        center_scale = scale * (1 - 1 / resolution)
        xy = torch.linspace(-center_scale, center_scale, resolution).cuda()
        xy = torch.meshgrid(xy, xy, indexing="xy")
        self.film_mask = -F.normalize(
            torch.stack(
                [xy[0], -xy[1], -torch.ones(resolution, resolution).cuda()], dim=2
            ),
            dim=2,
        )
        self.fov_mask = xy[0] ** 2 + xy[1] ** 2 <= scale * scale
        self.cos_mask = 1 / torch.sqrt(xy[0] ** 2 + xy[1] ** 2 + 1)

        self.hand_rot = (
            torch.tensor([[-0.1, 0.002, -0.05]]).cuda().repeat(hand_batch_size, 1)
        )
        # self.global_rot = (
        #     torch.tensor([[1, 0, 0], [0, 1, 0]])
        #     .float()
        #     .cuda()
        #     .repeat(hand_batch_size, 1)
        # )

        self.hand_batch_size = hand_batch_size
        self.camera_batch_size = hand_batch_size * self.num_cameras

        self.camera_sensor_distance = 0.001
        if not single_camera:
            angles = (
                (3.5 - torch.arange(0, self.num_cameras).repeat(hand_batch_size).cuda())
                * torch.pi
                * 2
                / self.num_cameras
            )
            self.sensor_centers = torch.stack(
                [
                    torch.zeros_like(angles),
                    self.radius * torch.cos(angles),
                    self.radius * torch.sin(angles),
                ]
            ).T
            self.sensor_norms = torch.stack(
                [
                    -np.sin(degree) * torch.ones_like(angles),
                    -np.cos(degree) * torch.cos(angles),
                    -np.cos(degree) * torch.sin(angles),
                ]
            ).T
            self.up = torch.tensor([[1, 0, 0]]).float().cuda()
        else:
            self.sensor_centers = torch.tensor([[0, 0, 0]]).float().cuda()
            self.sensor_norms = torch.tensor([[-1, 0, 0]]).float().cuda()
            self.up = torch.tensor([[0, 1, 0]]).float().cuda()

        self.num_bins = num_bins
        self.bin_size = bin_size

        self.mano_layer = ManoLayer(
            mano_assets_root="SPAD-Hand-Sim/data", flat_hand_mean=False
        ).cuda()
        
        self.hand_faces = self.mano_layer.th_faces.int()
        # load wrist mesh
        verts, faces, aux = load_obj(
            "SPAD-Hand-Sim/data/models/hand_01.obj",
            device="cuda",
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
        self.mano_layer.th_weights = torch.cat(
            [
                self.mano_layer.th_weights,
                self.mano_layer.th_weights[777:, :].repeat(64, 1),
            ]
        )
        self.mano_layer.th_J_regressor = torch.cat(
            [
                self.mano_layer.th_J_regressor,
                torch.zeros(16, 64).cuda(),
            ],
            dim=1,
        )
        self.mano_layer.th_posedirs = torch.cat(
            [
                self.mano_layer.th_posedirs,
                torch.zeros(64, 3, 135).cuda(),
            ]
        )
        self.mano_layer.wrist_verts = (
            verts.unsqueeze(0).repeat(hand_batch_size, 1, 1).cuda()
        )
        self.hand_faces = faces.verts_idx.int()
        self.mano_layer.root_indices = [
            78,
            79,
            38,
            92,
            108,
            117,
            118,
            119,
            120,
            234,
            239,
            122,
            121,
            214,
            215,
            79,
        ]

        # set up nvdiffrast
        self.ctx = dr.RasterizeCudaContext()
        n = 0.0001
        f = 1
        self.ndc_proj = (
            torch.tensor(
                [
                    [1 / scale, 0, 0, 0],
                    [0, -1 / scale, 0, 0],
                    [0, 0, -(f + n) / (f - n), -1],
                    [0, 0, -2 * f * n / (f - n), 0],
                ]
            )
            .float()
            .cuda()
        )

        # set up visualization
        self.camera_meshes = Mesh(
            vertices=self.sensor_centers.detach().cpu(), vc=colors["red"]
        )

        # set up jitter
        if not single_camera:
            jitters = np.load("SPAD-Hand-Sim/data/new_jitter.npz")
        else:
            jitters = np.load("SPAD-Hand-Sim/data/single_jitter.npz")
            
        self.jitter_pdf = torch.tensor(jitters["jitter_pdf"]).float().cuda()
        self.num_cycles = int(1500e3 * 9)
        # self.hist_offset = 9.523497581481934
        # self.bin_scaling = 0.2788732647895813
        self.jitter_kernel = (
            torch.tensor(jitters["jitter_kernel"])
            .float()
            .cuda()
            .flip(dims=(2,))
            .repeat(hand_batch_size, 1, 1)
        )

        # set up SPAD constants
        self.albedo = torch.tensor([0.2]).float().cuda().repeat(hand_batch_size, 1)
        
        self.noise = torch.tensor([1e-6]).float().cuda().repeat(hand_batch_size, 1)

        self.cube_mode = cube_mode
        if cube_mode:
            self.cube_vertices = (
                torch.tensor(
                    [
                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 1],
                            [1, 1, 1],
                            [0, 1, 1],
                        ]
                    ]
                )
                .float()
                .cuda()
            )
            self.cube_faces = (
                torch.tensor(
                    [
                        [0, 2, 1],
                        [2, 0, 3],
                        [1, 4, 0],
                        [4, 1, 5],
                        [3, 4, 7],
                        [4, 3, 0],
                        [6, 1, 2],
                        [1, 6, 5],
                        [4, 6, 7],
                        [6, 4, 5],
                        [3, 6, 2],
                        [6, 3, 7],
                    ]
                )
                .int()
                .cuda()
            )
            self.cube_normals = (
                torch.tensor(
                    [
                        [
                            [0, 0, 0],
                            [0, 0, -1],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 0],
                            [-1, 0, 0],
                            [-1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, 0],
                        ]
                    ]
                )
                .float()
                .cuda()
            )

    def update_sensor_transformations(self):
        z_axis = -(self.sensor_norms + torch.randn_like(self.sensor_norms) * 0.0002)
        x_axis = F.normalize(torch.cross(self.up, z_axis, dim=1), eps=1e-5)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
        self.R = torch.cat(
            (x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1
        ).transpose(1, 2)
        # camera is on the left side
        self.camera_vector = -self.camera_sensor_distance * x_axis
        self.T = -torch.bmm(
            self.R.transpose(1, 2),
            ((self.sensor_centers + torch.randn_like(self.sensor_centers) * 0.0002 )+ self.camera_vector).unsqueeze(-1),
        ).mT

    def forward(
        self,
        shape,
        pose,
        hand_transl=None,
        hand_rot=None,
        global_rot=None,
        image_path=None,
        random_sample=False,
        given_vertices=None
    ):
        """
        Args:
            - shape : Bx10
            - pose : Bx45
        """
        self.update_sensor_transformations()
        if hand_transl is None:
            hand_transl = self.hand_transl

        if hand_rot is not None:
            self.hand_rot = hand_rot
            
        if not self.cube_mode:
            pose = torch.cat([self.hand_rot, pose], dim=1)  
            vertices = self.mano_layer(pose, shape).verts + hand_transl[:, None, :]

            if given_vertices is not None:
                vertices = torch.cat([given_vertices.unsqueeze(0)[:,:778],vertices[:,778:]], dim=1)
            
            # import trimesh
            # mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy()[0], process=False)
            # mesh.export('origin_output_mesh.ply')
            # exit(0)
                
            # compute hand batch vertex normals BxVx3
            self.faces = self.hand_faces
            vertices_faces = vertices[:, self.faces]
            faces_normals = torch.cross(
                vertices_faces[:, :, 2, :] - vertices_faces[:, :, 1, :],
                vertices_faces[:, :, 0, :] - vertices_faces[:, :, 1, :],
                dim=2,
            )
            verts_normals = torch.zeros_like(vertices)
            verts_normals.index_add_(1, self.faces[:, 0], faces_normals)
            verts_normals.index_add_(1, self.faces[:, 1], faces_normals)
            verts_normals.index_add_(1, self.faces[:, 2], faces_normals)
            verts_normals = F.normalize(verts_normals, eps=1e-6, dim=2)
        else:
            b0 = torch.nn.functional.normalize(global_rot[0], dim=0)
            b1 = torch.nn.functional.normalize(
                global_rot[1] - b0.dot(global_rot[1]) * b0, dim=0
            )
            b2 = b0.cross(b1)
            so3 = torch.stack([b0, b1, b2])
            vertices = self.cube_vertices @ so3 + hand_transl
            self.faces = self.cube_faces

        self.vertices = vertices
        vertices = (
            torch.bmm(vertices.repeat_interleave(self.num_cameras, 0), self.R) + self.T
        )

        pos_clip = (
            torch.cat([vertices, torch.ones([*vertices.shape[:2], 1]).cuda()], dim=-1)
            @ self.ndc_proj
        )

        out, _ = dr.rasterize(
            self.ctx, pos_clip, self.faces, (self.resolution, self.resolution)
        )

        if not self.cube_mode:
            normals, _ = dr.interpolate(
                verts_normals.repeat_interleave(self.num_cameras, 0) @ self.R,
                out,
                self.faces,
            )
        else:
            faces_normals = torch.bmm(
                self.cube_normals.repeat_interleave(self.num_cameras, 0) @ so3, self.R
            )
            indices = out[:, :, :, 3].int()
            normals = torch.zeros_like(out[:, :, :, :3])
            for i in range(self.num_cameras):
                normals[i] = (
                    faces_normals[i]
                    .index_select(0, indices[i].reshape(256 * 256))
                    .reshape(256, 256, 3)
                )
        dots = (
            torch.einsum("ijk,bijk->bij", self.film_mask, normals).clip(0, 1)
            # .unsqueeze(-1)
            # .contiguous()
        )  # * self.fov_mask
        # dots = dr.antialias(dots, out, pos_clip, self.faces)[..., 0]
        # print(dots.shape)
        # dmin = dots.argmin()
        # db = dmin // (256 * 256)
        # di = (dmin // 256) % 256
        # dj = dmin % 256
        # print(db, di, dj, self.film_mask[di, dj], normals[db, di, dj])

        # dists = torch.linalg.norm(vertices, dim=2, keepdim=True)
        # depths = dr.interpolate(dists, out, self.faces)[0][..., 0]
        intersections = dr.interpolate(vertices, out, self.faces)[0]
        # laser is at -camera_vector
        laser_points = (
            intersections
            - 2
            * self.camera_sensor_distance
            * torch.tensor([1.0, 0.0, 0.0]).float().cuda()[None, None, None, :]
        )
        laser_distances = torch.linalg.norm(laser_points, dim=3)
        # the laser direction is the same as the camera viewing direction
        # laser_tan = laser_points[..., 0:2].abs().max(dim=3)[0] / laser_points[..., 2]
        laser_tan = (
            torch.sqrt(laser_points[..., 0] ** 2 + laser_points[..., 1] ** 2)
            / laser_points[..., 2]
        )
        sigma = np.tan(np.radians(21.5)) / 1.4 # np.tan(np.radians(21.5)) / 2.355
        laser_mask = torch.exp(-(laser_tan**2) / (2 * sigma * sigma))
        dots *= laser_mask

        # depths[depths == 0] = 1
        # albedo = self.albedo.repeat_interleave(self.num_cameras, 0).unsqueeze(-1)
        radiance = (
            dots
            # * albedo
            / laser_distances
            / laser_distances
            / torch.pi
            * self.width
            * self.width
            # * self.cos_mask
            * self.cos_mask
            * self.cos_mask
            * self.cos_mask
            / self.resolution
            / self.resolution
        )

        depths = (torch.linalg.norm(intersections, dim=3) + laser_distances) / 2
        d_bin = (depths / self.bin_size)[:, :, :, None].expand(
            *depths.shape, self.num_bins
        )
        # print(d_bin.shape)
        # transients = torch.zeros((self.camera_batch_size, self.num_bins))
        self.sigma = 3
        b = torch.arange(self.num_bins).cuda().expand(*d_bin.shape)
        # kernels = torch.exp(-(((d_bin - b - 0.5) / self.sigma) ** 2) / 2) / (
        #     self.sigma * np.sqrt(2.0 * np.pi)
        # )
        kernels = torch.sigmoid(self.sigma * (d_bin - b)) - torch.sigmoid(
            self.sigma * (d_bin - b - 1)
        )
        transients = torch.einsum("bijk,bij->bk", kernels, radiance)

        if image_path is not None:
            render = (
                dots.unsqueeze(-1) * torch.tensor([0.96, 0.75, 0.69]).cuda()
            ).permute(0, 3, 1, 2)
            torchvision.utils.save_image(render, image_path)
            depth_map = (
                (1 - depths / self.bin_size / 16).unsqueeze(-1)
                * torch.tensor([1, 1, 1]).cuda()
            ).permute(0, 3, 1, 2)
            torchvision.utils.save_image(depth_map, "SPAD-Hand-Sim/results/depths.png")
            self.hand_mesh = hand_meshes(self.vertices, self.faces)[0]
            # hj_meshes = Mesh.concatenate_meshes([self.hand_mesh, self.camera_meshes])
            import trimesh
            concatenated_mesh = trimesh.util.concatenate([self.hand_mesh, self.camera_meshes])
            concatenated_mesh.export('SPAD-Hand-Sim/results/hand_camera_mesh.ply')
            print("save mesh")
            # hj_meshes.show()

        distance = -hand_transl[0, 0] - 1
        # print(radiance.sum(), transients.sum())
        self.transients = transients
        # print(transients[0])
        return self.simulate_sensor_response(sample=random_sample)

    def simulate_sensor_response(self, sample=True):
        hists = self.transients * (self.albedo + torch.randn_like(self.albedo).cuda() * 0.002 ) + self.noise

        r_sum = torch.cumsum(
            torch.concatenate(
                (torch.zeros((len(hists), 1)).cuda(), hists[:, :-1]), axis=1
            ),
            axis=1,
        )
        p = (1 - torch.exp(-hists)) * torch.exp(-r_sum)
        p = torch.concatenate(
            (p, torch.clip(1 - p.sum(axis=1, keepdims=True), 0, 1)), axis=1
        )
        if sample:
            with torch.no_grad():
                for i in range(len(hists)):
                    samples = torch.multinomial(p[i], self.num_cycles, replacement=True)
                    samples = samples[samples < self.num_bins]
                    num_hit_photons = len(samples)
                    # print(num_hit_photons)
                    if num_hit_photons == 0:
                        hists[i] = torch.zeros_like(hists[i])
                        continue
                    hists[i] = torch.histc(
                        samples + self.hist_offset
                        # + torch.rand(num_hit_photons).cuda()
                        + torch.multinomial(
                            self.jitter_pdf[i % 8],
                            num_hit_photons,
                            replacement=True,
                        )
                        * self.bin_scaling,
                        bins=self.num_bins,
                        min=0,
                        max=self.num_bins - 1,
                    )
                hists /= self.num_cycles
                # hists = hists + 0.5 * hists**2
                # hists /= hists.sum(axis=1, keepdims=True)
            return hists
        else:
            torch.backends.cudnn.deterministic = True
            hit_rates = 1 - p[:, -1].detach().cpu().numpy()
            # print(hit_rates * 36e6)
            q = F.conv1d(
                p[:, :-1].reshape(
                    len(p) // self.num_cameras, self.num_cameras, self.num_bins
                ),
                self.jitter_kernel,
                groups=self.num_cameras,
                padding="same",
            )[0]
            # q = q + 0.5 * q**2
            # q /= hists.sum(axis=1, keepdims=True)

            return q
