import numpy as np
import trimesh

def sample_mesh(mesh, num_samples, mode="uniform"):
    """
    Draw surface points from mesh.
    Code adapted from https://github.com/mikedh/trimesh/blob/master/trimesh/sample.py

    Args:
        mesh (Trimesh): mesh object.
        num_samples (int): total number of surface point samples.

    Returns:
        samples (float array, (n, 3)): surface points in world frame.
        face_id (int array, (n, )): face ID of a surface point.
        weights (float array, (n, )): importance weight of a surface point.
    """

    # uniform sampling
    samples = np.random.random(num_samples) * mesh.area
    face_id = np.searchsorted(np.cumsum(mesh.area_faces), samples)
    weights = mesh.area / num_samples * np.ones(num_samples)

    # represent triangles in the form of origin + 2 vectors
    tri_origins = mesh.triangles[face_id, :1]
    tri_vectors = mesh.triangles[face_id, 1:] - tri_origins

    # scale the vectors by coefficients randomly sampled from [0, 1]
    lengths = np.random.random((len(tri_vectors), 2, 1))
    ## NOTE: The points will be uniformly distributed on a quadrilateral.
    ## A point will be inside the triangle if the scalars sum less than 1.
    ## We transform vectors longer than 1 to be inside the triangle by
    ## "wrapping around" the scalar values.
    test = np.sum(lengths, axis=1).squeeze(1) > 1.0  # out-of-bound vectors
    lengths[test] -= 1.0
    lengths = np.abs(lengths)

    # generate sample points
    sample_vectors = np.sum(tri_vectors * lengths, axis=1)
    samples = tri_origins.squeeze(1) + sample_vectors
    return samples, face_id, weights


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
        phi = 2 * np.pi * np.random.random(num_samples)
        sinp, cosp = np.sin(phi), np.cos(phi)
        cost = np.random.uniform(np.cos(anchor.fov / 2), 1.0, num_samples)
        sint = np.sqrt(1 - cost**2)
        weights = anchor.solid_angle / num_samples * np.ones(num_samples)
    elif mode == "cosine":  # cosine-weighted sampling
        phi = 2 * np.pi * np.random.random(num_samples)
        sina = np.sin(anchor.fov / 2)
        sinp, cosp = np.sin(phi), np.cos(phi)
        sint = np.sqrt(np.random.random(num_samples)) * sina
        cost = np.sqrt(1 - sint**2)
        weights = np.pi * sina**2 / (num_samples * cost)
    else:
        raise NotImplementedError(f"sampling mode [{mode}] not supported")

    samples = np.vstack((sint * cosp, sint * sinp, cost)).T

    # rotate from canonical pose to align with cone axis
    samples = np.dot(samples, anchor.a2w[:3, :3].T)
    return samples, weights


def interpolate_normals(mesh, face_id, barycoords):
    """
    Barycentric interpolation of vertex normals.

    Args:
        mesh (Trimesh): mesh object.
        face_id (int array, (n, )): face index.
        barycoords (float array, (n, 3)): barycentric coordinates.

    Returns:
        n (float array, (n, 3)): interpolated normals.
    """
    assert len(face_id) == len(barycoords)

    vn = mesh.vertex_normals[mesh.faces[face_id]]
    n = np.sum(np.expand_dims(barycoords, axis=2) * vn, axis=1)
    return n


def evaluate_path_integral(
    mesh,  # mesh object
    light,  # light source object
    camera,  # camera object
    albedo,  # spatially-uniform albedo
    num_samples,  # total number of point samples
    normal_mode="fn",  # face ("fn") or vertex normals ("vn")
):
    """Path integral form of transient light transport."""

    assert albedo >= 0 and albedo <= 1, "albedo must be between 0 and 1"
    fr = albedo / np.pi  # spatially uniform Lambertian BRDF

    # surface sampling
    x, fid, w = sample_mesh(mesh, num_samples)
    print(x.shape)
    print(fid.shape)
    print("----------------------------")

    # ray-mesh intersection (light >> mesh)
    r1_origins = np.tile(light.loc, (len(x), 1))
    print("light loc is:", light.loc)
    print("r1_origins shape is ", r1_origins.shape)
    
    r1_dirs = x - r1_origins
    hit_fid, rid = mesh.ray.intersects_id(r1_origins, r1_dirs, return_locations=False, multiple_hits=False)
    
    x, fid, w = x[rid], fid[rid], w[rid]
    r1_dirs = r1_dirs[rid]
    print(x.shape)
    print(fid.shape)
    print("----------------------------")

    # filter out occluded surface points
    assert len(fid) == len(hit_fid)
    vis_id = np.where(fid == hit_fid)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    r1_dirs = r1_dirs[vis_id]
    print(x.shape)
    print(fid.shape)
    print("----------------------------")

    # ray-mesh intersection (camera >> mesh)
    r2_origins = np.tile(camera.loc, (len(x), 1))
    print("camera loc is:", camera.loc)
    r2_dirs = x - r2_origins
    hit_fid, rid = mesh.ray.intersects_id(  # hit_fid is face id
        r2_origins, r2_dirs, return_locations=False, multiple_hits=False
    )
    print("origins are: ", r2_origins.shape)
    print("dirs are: ", r2_dirs.shape)
    print("hit_fid shape is: ", hit_fid.shape)
    print("hit_fid is: ", hit_fid)
    print("rid shape is: ", rid.shape)
    print("rid is: ", rid)

    x, fid, w = x[rid], fid[rid], w[rid]
    r1_dirs, r2_dirs = r1_dirs[rid], r2_dirs[rid]
    print("----------------------------")
    print(x.shape)
    print(fid.shape)
    print("r1 dirs are: ", r2_dirs.shape)
    print("r2 dirs are: ", r2_dirs.shape)

    ###################################################################################################

    # filter out occluded surface points
    assert len(fid) == len(hit_fid)
    vis_id = np.where(fid == hit_fid)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    r1_dirs, r2_dirs = r1_dirs[vis_id], r2_dirs[vis_id]

    # ray segment lengths and normalized directions
    d1 = np.linalg.norm(r1_dirs, axis=1)
    d2 = np.linalg.norm(r2_dirs, axis=1)
    v1 = r1_dirs / np.expand_dims(d1, axis=1)
    v2 = r2_dirs / np.expand_dims(d2, axis=1)

    # filter out rays outside of light cone
    nl = np.tile(light.normal, (len(x), 1))
    cosl = np.einsum("ij,ij->i", nl, v1)
    cosl = np.clip(cosl, 0.0, 1.0)
    rl_angle = np.arccos(cosl)
    vis_id = np.where(np.logical_and(rl_angle >= 0, rl_angle <= light.fov / 2))[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    d1, v1 = d1[vis_id], v1[vis_id]
    d2, v2 = d2[vis_id], v2[vis_id]

    # filter out rays outside of camera field-of-view
    nc = np.tile(camera.normal, (len(x), 1))
    cosc = np.einsum("ij,ij->i", nc, v2)
    cosc = np.clip(cosc, 0.0, 1.0)
    rs_angle = np.arccos(cosc)
    vis_id = np.where(np.logical_and(rs_angle >= 0, rs_angle <= camera.fov / 2))[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    d1, v1 = d1[vis_id], v1[vis_id]
    d2, v2 = d2[vis_id], v2[vis_id]
    cosc = cosc[vis_id] # to filter out large angle

    # calculate surface normals for intersecting points
    if normal_mode == "fn":  # use face normals
        n = mesh.face_normals[fid]
    elif normal_mode == "vn":  # interpolate vertex normals
        bary_x = trimesh.triangles.points_to_barycentric(mesh.triangles[fid], x)
        n = interpolate_normals(mesh, fid, bary_x)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    else:
        raise NotImplementedError(f"normal mode [{normal_mode}] not supported")

    # calculate geometry term (cosine foreshortening + quadratic fall-off)
    cos1 = np.einsum("ij,ij->i", n, -v1)
    cos2 = np.einsum("ij,ij->i", n, -v2)
    cos1 = np.clip(cos1, 0.0, 1.0)
    cos2 = np.clip(cos2, 0.0, 1.0)
    geometry = (cos1 * cos2) / (d1**2 * d2**2)

    if light.is_area:  # area light
        geometry *= cosl
    if camera.is_area:  # area camera
        geometry *= cosc

    # perform distance binning
    d_bin = np.floor((d1 + d2) / camera.bin_size)
    d_idx = np.where(d_bin < camera.num_bins)[0]

    # calculate radiance contribution of each ray to the transient
    radiance = light.power * fr * geometry * w

    # populate the transient
    transient = np.zeros(camera.num_bins)
    bins = np.unique(d_bin[d_idx])  # bins with non-zero irradiance
    for b in bins:
        transient[int(b)] += np.sum(radiance[d_bin == b])  # per-bin irradiance
    return transient


def evaluate_angular_integral(
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
    fr = albedo / np.pi  # spatially uniform Lambertian BRDF

    # directional sampling
    if anchor == "light":  # sample light cone
        source, target = light, camera
    elif anchor == "camera":  # sample camera cone
        source, target = camera, light
    else:
        raise NotImplementedError(f"anchor type [{anchor}] not supported")

    omega, w = sample_direction(source, num_samples, sample_mode)
    num_samples = len(omega)

    # ray-mesh intersection (source >> mesh)
    r1_origins = np.tile(source.loc, (len(omega), 1))
    fid, rid, x = mesh.ray.intersects_id(r1_origins, omega, return_locations=True, multiple_hits=False)
    r1_dirs = x - r1_origins[rid]
    w = w[rid]

    # ray-mesh intersection (target >> mesh)
    r2_origins = np.tile(target.loc, (len(x), 1))
    r2_dirs = x - r2_origins
    hit_fid, rid = mesh.ray.intersects_id(r2_origins, r2_dirs, return_locations=False, multiple_hits=False)
    x, fid, w = x[rid], fid[rid], w[rid]
    r1_dirs, r2_dirs = r1_dirs[rid], r2_dirs[rid]

    # filter out occluded surface point samples
    assert len(fid) == len(hit_fid)
    vis_id = np.where(fid == hit_fid)[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    r1_dirs, r2_dirs = r1_dirs[vis_id], r2_dirs[vis_id]

    # ray segment lengths and normalized directions
    d1 = np.linalg.norm(r1_dirs, axis=1)  # camera->mesh; mesh->sensor, should be same in our setting
    d2 = np.linalg.norm(r2_dirs, axis=1)
    v1 = r1_dirs / np.expand_dims(d1, axis=1)
    v2 = r2_dirs / np.expand_dims(d2, axis=1)

    # filter out rays outside of target cone
    nt = np.tile(target.normal, (len(x), 1))
    cost = np.einsum("ij,ij->i", nt, v2)
    cost = np.clip(cost, 0.0, 1.0)
    rt_angle = np.arccos(cost)
    vis_id = np.where(np.logical_and(rt_angle >= 0, rt_angle <= target.fov / 2))[0]
    x, fid, w = x[vis_id], fid[vis_id], w[vis_id]
    d1, v1 = d1[vis_id], v1[vis_id]
    d2, v2 = d2[vis_id], v2[vis_id]

    # calculate surface normals for intersecting points
    if normal_mode == "fn":  # use face normals
        n = mesh.face_normals[fid]
    elif normal_mode == "vn":  # interpolate vertex normals
        bary_x = trimesh.triangles.points_to_barycentric(mesh.triangles[fid], x)
        n = interpolate_normals(mesh, fid, bary_x)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    else:
        raise NotImplementedError(f"normal mode [{normal_mode}] not supported")

    # calculate geometry term
    cos2 = np.einsum("ij,ij->i", n, -v2)
    cos2 = np.clip(cos2, 0.0, 1.0)
    geometry = cos2 / d2**2

    if source.is_area:  # area source
        ns = np.tile(source.normal, (len(x), 1))
        coss = np.einsum("ij,ij->i", ns, v1)
        coss = np.clip(coss, 0.0, 1.0)
        geometry *= coss
    if target.is_area:  # area target
        geometry *= cost

    # perform distance binning
    d_bin = np.floor((d1 + d2) / camera.bin_size)
    d_idx = np.where(d_bin < camera.num_bins)[0]

    # calculate radiance contribution of each ray to the transient
    radiance = light.power * fr * geometry * w

    # populate the transient
    transient = np.zeros(camera.num_bins)
    bins = np.unique(d_bin[d_idx])  # bins with non-zero irradiance
    for b in bins:
        transient[int(b)] += np.sum(radiance[d_bin == b])  # per-bin irradiance
    return transient
