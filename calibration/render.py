import torch
import numpy as np
import trimesh

torch.autograd.set_detect_anomaly(True)

def sample_mesh(vertices, mesh, num_samples):
    """
    Draw surface points from mesh in PyTorch. This function assumes that mesh data is accessible
    as tensors and that the cumulative sum and other operations respect gradient flow.
    """
    
    faces = torch.from_numpy(mesh.faces).long()
    vertices = vertices.float()

    # Sample areas proportionally and get face indices
    samples = torch.rand(num_samples) * mesh.area
    face_id = torch.searchsorted(torch.cumsum(torch.from_numpy(mesh.area_faces), dim=0), samples)

    weights = mesh.area / num_samples * torch.ones(num_samples, device=samples.device)
    
    # Get the vertex positions for the sampled faces
    tri_vertices = vertices[faces[face_id]]

    # Calculate triangle origins and vectors
    tri_origins = tri_vertices[:, 0, :]
    tri_vectors = tri_vertices[:, 1:] - tri_origins.unsqueeze(1)
    
    # Generate random coefficients for linear combination inside the triangle
    lengths = torch.rand((len(tri_vectors), 2, 1))

    test = lengths.sum(dim=1).squeeze(1) > 1.0  # out-of-bound vectors
    lengths[test] -= 1.0
    lengths = torch.abs(lengths)

    # generate sample points
    sample_vectors = (tri_vectors * lengths).sum(dim=1)
    samples = tri_origins.squeeze(1) + sample_vectors
    
    return samples, face_id, weights


def interpolate_normals(mesh, face_id, barycoords):
    """
    Barycentric interpolation of vertex normals using PyTorch.

    Args:
        mesh (Trimesh-like object): mesh object with attributes 'vertex_normals' and 'faces', both should be PyTorch tensors.
        face_id (torch.Tensor, (n,)): face index, should be of type long.
        barycoords (torch.Tensor, (n, 3)): barycentric coordinates.

    Returns:
        n (torch.Tensor, (n, 3)): interpolated normals.
    """
    assert face_id.shape[0] == barycoords.shape[0], "face_id and barycoords must have the same length"

    # Index into vertex_normals using indices from faces and face_id
    vn = torch.from_numpy(mesh.vertex_normals[mesh.faces[face_id]])

    # Barycentric interpolation
    n = torch.sum(barycoords.unsqueeze(2) * vn, dim=1)

    return n

def compute_face_normals(vertices, faces):
    """
    Compute face normals in a differentiable way using vertex positions.

    Args:
        vertices (torch.Tensor): Tensor of vertex positions, shape [num_vertices, 3]
        faces (torch.Tensor): Tensor of indices into vertices, shape [num_faces, 3]

    Returns:
        torch.Tensor: Normalized face normals, shape [num_faces, 3]
    """
    # Gather vertices for each face
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute normals as cross product of edges
    normals = torch.cross(edge1, edge2, dim=1)

    # Normalize the normals
    normals = torch.nn.functional.normalize(normals, p=2, dim=1)

    return normals


def evaluate_path_integral(
    vertices,  # vertices with gradient
    mesh,   # trimesh object
    light,  # light source object with properties as tensors
    camera,  # camera object with properties as tensors
    albedo,  # spatially-uniform albedo as a tensor
    num_samples,  # total number of point samples
    normal_mode="fn",  # face ("fn") or vertex normals ("vn")
):
    """Path integral form of transient light transport in PyTorch."""

    assert albedo >= 0 and albedo <= 1, "albedo must be between 0 and 1"
    fr = albedo / torch.pi  # spatially uniform Lambertian BRDF

    # surface sampling
    x, fid, w = sample_mesh(vertices, mesh, num_samples) # x shape: [500000, 3]
    
    # ray-mesh intersection (light >> mesh)
    r1_origins = (light.loc.unsqueeze(0)).repeat(len(x), 1) # shape: [500000, 3]
    r1_dirs = x - r1_origins

    hit_fid, rid = mesh.ray.intersects_id(r1_origins.clone().detach().numpy(), r1_dirs.clone().detach().numpy() , return_locations=False, multiple_hits=False)
    x, fid, w = x[rid], fid[rid], w[rid]
    r1_dirs = r1_dirs[rid]
    hit_fid = torch.from_numpy(hit_fid)

    # Filter logic similar to above; using torch.where, torch.norm, etc.
    vis_id = torch.where(fid == hit_fid)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    r1_dirs = r1_dirs[vis_id]
    
    # ray-mesh intersection (camera >> mesh)
    r2_origins = (camera.loc.unsqueeze(0)).repeat(len(x), 1)
    r2_dirs = x - r2_origins
    hit_fid, rid = mesh.ray.intersects_id(  # hit_fid is face id
        r2_origins.clone().detach().numpy(), r2_dirs.clone().detach().numpy(), return_locations=False, multiple_hits=False
    )
    x, fid, w = x[rid], fid[rid], w[rid]
    r1_dirs, r2_dirs = r1_dirs[rid], r2_dirs[rid]
    hit_fid = torch.from_numpy(hit_fid)
    
    ###################################################################################################
    
    # filter out occluded surface points
    assert len(fid) == len(hit_fid)
    vis_id = torch.where(fid == hit_fid)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    r1_dirs, r2_dirs = r1_dirs[vis_id], r2_dirs[vis_id]

    # ray segment lengths and normalized directions
    d1 = torch.norm(r1_dirs, dim=1)
    d2 = torch.norm(r2_dirs, dim=1)
    v1 = (r1_dirs / d1.unsqueeze(1)).to(dtype=torch.float32)
    v2 = (r2_dirs / d2.unsqueeze(1)).to(dtype=torch.float32)

    # filter out rays outside of light cone
    nl = (light.normal.unsqueeze(0)).repeat(len(x), 1)
    cosl = torch.einsum("ij,ij->i", nl, v1)
    cosl = torch.clamp(cosl, 0.0, 1.0)
    rl_angle = torch.acos(cosl)
    vis_id = torch.where((rl_angle >= 0) & (rl_angle <= light.fov / 2))[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    d1, v1 = d1[vis_id], v1[vis_id]
    d2, v2 = d2[vis_id], v2[vis_id]

    # filter out rays outside of camera field-of-view
    nc = (camera.normal.unsqueeze(0)).repeat(len(x), 1)
    cosc = torch.einsum("ij,ij->i", nc, v2)
    cosc = torch.clamp(cosc, 0.0, 1.0)
    rs_angle = torch.acos(cosc)
    vis_id = torch.where((rs_angle >= 0) & (rs_angle <= camera.fov / 2))[0]
    
    x_all, fid_all, w_all = x, fid, w # store all rays information
    cosl_all = cosl # store all rays information
    d1_all, v1_all, d2_all, v2_all = d1, v1, d2, v2 # store all rays information
    cosc_all = cosc # store all rays information
    
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    d1, v1 = d1[vis_id], v1[vis_id]
    d2, v2 = d2[vis_id], v2[vis_id]
    cosc = cosc[vis_id] # to filter out large angle
    
    # Calculate normals based on the mode
    if normal_mode == "fn":
        # n = get_face_normals_torch(mesh, fid)  # Needs implementation   
        n = compute_face_normals(mesh.vertices, mesh.faces)
    elif normal_mode == "vn":  # interpolate vertex normals
        bary_x = trimesh.triangles.points_to_barycentric(mesh.triangles[fid], x.clone().cpu().detach().numpy())
        n = interpolate_normals(mesh, fid, torch.from_numpy(bary_x))
        n /= torch.norm(n, dim=1, keepdim=True)
        # n = compute_face_normals(vertices, mesh.faces)
    else:
        n = compute_face_normals(mesh.vertices, mesh.faces)
    # elif normal_mode == "vn":
    #     bary_x = points_to_barycentric_torch(mesh.triangles[fid], x)  # Needs implementation
    #     n = interpolate_normals_torch(mesh, fid, bary_x)  # Needs implementation

    # Geometry and radiance calculations
    # print("n size, ", n.size())
    # print("v1 size, ", v1.size())
    cos1 = torch.einsum("ij,ij->i", n, -v1.to(dtype=torch.float64))
    cos2 = torch.einsum("ij,ij->i", n, -v2.to(dtype=torch.float64))
    cos1 = torch.clamp(cos1, 0.0, 1.0)
    cos2 = torch.clamp(cos2, 0.0, 1.0)
    geometry = (cos1 * cos2) / (d1**2 * d2**2) # hard to optimize when (d1**2 * d2**2), so use angular_integral

    if light.is_area:  # area light
        geometry = geometry*cosl
    if camera.is_area:  # area camera
        geometry = geometry*cosc

    # perform distance binning
    d_bin = torch.floor((d1 + d2) / camera.bin_size)
    d_idx = torch.where(d_bin < camera.num_bins)[0]
    
    # Continue with PyTorch tensor operations for distance binning and radiance calculation
    radiance = light.power * fr * geometry * w
    
    radiance = radiance.to(torch.float32)
    
    transient = torch.zeros(camera.num_bins, device=d_bin.device, dtype=radiance.dtype)

    # Create a mask for valid bins
    valid_mask = (d_bin >= 0) & (d_bin < camera.num_bins)

    # Filter bins and radiance values using the valid mask
    valid_bins = d_bin[valid_mask].to(torch.float32)
    valid_radiance = radiance[valid_mask]

    # Accumulate values into transient using a loop
    for idx in range(camera.num_bins):
        # Create a mask for each bin
        bin_mask = (valid_bins == idx)
        bin_outmask = (valid_bins != idx)
        # Sum up radiance values where the mask is True
        transient[idx] =  0.9 * (1.0 * valid_radiance[bin_mask].sum() + 0.0 * valid_radiance[bin_outmask].sum()) 
    
    return transient 

def evaluate_angular_integral(
    vertices,
    mesh,  # mesh object
    light,  # light source object
    camera,  # camera object
    albedo,  # spatially-uniform albedo
    num_samples,  # total number of ray samples
    anchor="light",  # anchor type ("light" | "camera")
    sample_mode="uniform",  # uniform ("uniform") or cosine-weighted ("cosine") sampling
    normal_mode="fn",  # face ("fn") or vertex normals ("vn")
):
    """Angular integral form of transient light transport."""

    assert albedo >= 0 and albedo <= 1, "albedo must be between 0 and 1"
    fr = albedo / torch.pi  # spatially uniform Lambertian BRDF

    # directional sampling
    if anchor == "light":  # sample light cone
        source, target = light, camera
    elif anchor == "camera":  # sample camera cone
        source, target = camera, light
    else:
        raise NotImplementedError(f"anchor type [{anchor}] not supported")

    omega, w = sample_direction(source, num_samples, "large_fov")
    num_samples = len(omega)

    # ray-mesh intersection (source >> mesh)
    r1_origins = (source.loc.unsqueeze(0)).repeat(len(omega), 1)
    fid, rid, x = mesh.ray.intersects_id(r1_origins.clone().detach().numpy(), omega.clone().detach().numpy(), return_locations=True, multiple_hits=False)
    fid, rid, x = torch.from_numpy(fid),torch.from_numpy(rid), torch.from_numpy(x)
    
    r1_dirs = x - r1_origins[rid]
    w = w[rid]

    # ray-mesh intersection (target >> mesh)
    r2_origins = (target.loc.unsqueeze(0)).repeat(len(x), 1)
    r2_dirs = x - r2_origins
    hit_fid, rid = mesh.ray.intersects_id(r2_origins.clone().detach().numpy(), r2_dirs.clone().detach().numpy(), return_locations=False, multiple_hits=False)
    hit_fid, rid = torch.from_numpy(hit_fid),torch.from_numpy(rid)
    x, fid, w = x[rid], fid[rid], w[rid]
    r1_dirs, r2_dirs = r1_dirs[rid], r2_dirs[rid]

    ###################################################################################################
    
    # filter out occluded surface point samples
    assert len(fid) == len(hit_fid)
    vis_id = torch.where(fid == hit_fid)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    r1_dirs, r2_dirs = r1_dirs[vis_id], r2_dirs[vis_id]

    # ray segment lengths and normalized directions
    d1 = torch.norm(r1_dirs, dim=1)  # camera->mesh; mesh->sensor, should be same in our setting
    d2 = torch.norm(r2_dirs, dim=1)
    v1 = (r1_dirs / d1.unsqueeze(1)).to(dtype=torch.float32)
    v2 = (r2_dirs / d2.unsqueeze(1)).to(dtype=torch.float32)

    # filter out rays outside of target cone
    nt = (target.normal.unsqueeze(0)).repeat(len(x), 1)
    cost = torch.einsum("ij,ij->i", nt, v2)
    cost = torch.clamp(cost, 0.0, 1.0)
    rt_angle = torch.acos(cost)
    # vis_id = torch.where((rt_angle >= 0) & (rt_angle <= target.fov / 2))[0]
    vis_id = torch.where(rt_angle >= 0)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    d1, v1 = d1[vis_id], v1[vis_id]
    d2, v2 = d2[vis_id], v2[vis_id]
    cost = cost[vis_id]
    rt_angle = rt_angle[vis_id]

    # normal_mode = "fn"
    # calculate surface normals for intersecting points
    if normal_mode == "fn":  # use face normals
        n = compute_face_normals(mesh.vertices, mesh.faces)
    elif normal_mode == "vn":  # interpolate vertex normals
        bary_x = trimesh.triangles.points_to_barycentric(mesh.triangles[fid], x.clone().cpu().detach().numpy())
        n = interpolate_normals(mesh, fid, torch.from_numpy(bary_x))
        n = n/torch.norm(n, dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"normal mode [{normal_mode}] not supported")

    # calculate geometry term
    cos2 = torch.einsum("ij,ij->i", n.to(torch.float32), -v2.to(torch.float32))
    cos2 = torch.clamp(cos2, 0.0, 1.0)
    geometry = (cos2 / d2**2)

    if source.is_area:  # area source
        ns = (source.normal.unsqueeze(0)).repeat(len(x), 1)
        coss = torch.einsum("ij,ij->i", ns, v1)
        coss = torch.clamp(coss, 0.0, 1.0)
        geometry = geometry*coss
       
    if target.is_area:  # area target
        geometry = geometry*cost

    # perform distance binning
    d_bin = torch.floor((d1 + d2) / camera.bin_size)
    # d_idx = torch.where(d_bin < camera.num_bins)[0]

    # calculate radiance contribution of each ray to the transient
    radiance = light.power * fr * geometry * w
   
    ###############################################################################################
    # populate the transient
    transient = torch.zeros(camera.num_bins, device=d_bin.device, dtype=radiance.dtype)

    # Accumulate values into transient using a loop
    for idx in range(camera.num_bins):

        bin_c = torch.tensor(0.01)
        bin_sigma = torch.tensor(0.05)
        d0 = torch.tensor((0.5+idx) * camera.bin_size)
        # print("d0 is ", d0)
        # print("d1+d2 is ", d1+d2)
        bin_exponent = -(((d1+d2)/bin_c - d0/bin_c) ** 2) / (2 * bin_sigma ** 2)
        bin_coefficient = 1 / torch.sqrt(2 * torch.pi * bin_sigma ** 2)
        bin_weight = bin_coefficient * torch.exp(bin_exponent)

        bin_radiance = radiance * bin_weight
        transient[idx] =  bin_radiance.sum()

    return transient


def sample_direction(anchor, num_samples, mode="uniform"):
    """
    Draw ray directions from light or camera field of view.

    Args:
        anchor (Anchor): light source or camera object.
        num_samples (int): total number of ray samples.
        mode (str): uniform ("uniform") or cosine-weighted sampling ("cosine").

    Returns:
        samples (float array, (n, 3)): unit directions in world frame.
        weights (float array, (n, )): importance weights.
    """
    if mode == "uniform":  # uniform sampling
        phi = 2 * torch.pi * torch.rand(num_samples)
        sinp, cosp = torch.sin(phi), torch.cos(phi)
        cost = torch.rand(num_samples) * (1.0 - torch.cos(anchor.fov/ 2)) + torch.cos(anchor.fov / 2)
        sint = torch.sqrt(1 - cost**2)
        weights = anchor.solid_angle / num_samples * torch.ones(num_samples)
    elif mode == "cosine":  # cosine-weighted sampling
        phi = 2 * torch.pi * torch.rand(num_samples)
        sina = torch.sin(anchor.fov / 2)
        sinp, cosp = torch.sin(phi), torch.cos(phi)
        sint = torch.sqrt(torch.rand(num_samples)) * sina
        cost = torch.sqrt(1 - sint**2)
        weights = torch.pi * sina**2 / (num_samples * cost)
    elif mode == "large_fov":
        phi = 2 * torch.pi * torch.rand(num_samples)
        sinp, cosp = torch.sin(phi), torch.cos(phi)
        # cost = torch.rand(num_samples) * (1.0 - torch.cos(anchor.fov/ 2)) + torch.cos(anchor.fov / 2)
        cost = biased_sampling(num_samples)
        sint = torch.sqrt(1 - cost**2)
        # weights = anchor.solid_angle / num_samples * torch.ones(num_samples) # uniform weight
        weights = cost * 4.5e-7
    else:
        raise NotImplementedError(f"sampling mode [{mode}] not supported")

    samples = torch.vstack((sint * cosp, sint * sinp, cost)).T
    # rotate from canonical pose to align with cone axis
    samples = torch.matmul(samples, anchor.a2w[:3, :3].T)
    return samples, weights


def biased_sampling(num_samples, bias_threshold=0.9659, bias_ratio=0.9):
    """
    Sample variables in range [0, 1] with over 90% of the samples larger than 0.9659.

    Args:
        num_samples (int): Number of samples to generate.
        bias_threshold (float): The threshold value above which we want more samples.
        bias_ratio (float): The ratio of samples that should be above the bias_threshold.

    Returns:
        torch.Tensor: Sampled variables.
    """
    num_biased_samples = int(num_samples * bias_ratio)
    num_uniform_samples = num_samples - num_biased_samples
    biased_samples = torch.rand(num_biased_samples) * (1 - bias_threshold) + bias_threshold
    uniform_samples = torch.rand(num_uniform_samples) * bias_threshold
    samples = torch.cat((biased_samples, uniform_samples))
    samples = samples[torch.randperm(num_samples)]

    return samples
