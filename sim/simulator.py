import os
import time

import numpy as np
import trimesh

from .render import evaluate_path_integral, evaluate_angular_integral  # to .render
from .structs import Anchor, Light, Camera  # to .structs


def simulate_impulse_response(opt):
    # load mesh
    # mesh sits on xy-plane of a right-handed coordinate system
    assert os.path.exists(opt["mesh_path"])
    mesh = trimesh.load_mesh(opt["mesh_path"])

    # set up light sources and cameras
    # light source and camera are co-located with same orientation
    # cameras lie on the upper hemisphere centered at the origin
    setup_method = opt["setup_method"]
    anchor = Anchor(
        loc=[0, 0, 0],
        normal=[0, 0, 1],
        fov=180 - 2 * opt["setup"]["min_elevation"],
    )
    if setup_method == "fibonacci":
        # uniform sampling on Fibonacci spiral
        d = anchor.sample_fibonacci(opt["setup"]["num_cameras"])
    elif setup_method == "circle":
        # uniform sampling of circle coordinates
        d = anchor.sample_circle(opt["setup"]["num_cameras"])
    else:
        # uniform sampling of spherical coordinates
        d = anchor.sample_spherical(opt["setup"]["num_cameras"])
    locs = d * opt["setup"]["radius"]  # d is a num_sample*3 array
    if setup_method == "circle":
        normals = -d 
        normals[:,2] = 2.747
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]

    else:
        normals = -d
    
    lights, cameras = [], []
    loc_save, normal_save = [], []
    cam_count = 0 
    for loc, normal in zip(locs, normals):  # zip is to set correspondance
        light = Light(
            loc,
            normal,
            fov=opt["light"]["fov"],
            power=opt["light"]["power"],
            is_area=opt["light"]["is_area"],
        )
        camera = Camera(
            loc,
            normal,
            fov=opt["camera"]["fov"],
            bin_size=opt["camera"]["bin_size"],
            num_bins=opt["camera"]["num_bins"],
            is_area=opt["camera"]["is_area"],
        )
        lights.append(light)
        cameras.append(camera)
        
        loc_save.append(loc)
        normal_save.append(normal)
        cam_count = cam_count + 1

    # select rendering method
    # path integral: sample points on object surface
    # angular integral: sample rays from within light / camera FoV
    render_fn = evaluate_path_integral if opt["render_method"] == "path" else evaluate_angular_integral

    # run simulation
    print("Simulation started...")
    hists, Rts = [], []
    for idx, (light, camera) in enumerate(zip(lights, cameras)):
      
        hist = render_fn(mesh, light, camera, **opt["render"])
        
        hists.append(hist)
        Rts.append(camera.a2w)
        print(f"\t{(idx + 1):0{len(str(len(cameras)))}d} done.")
        
    hists = np.array(hists).astype(np.float32)
    Rts = np.array(Rts).astype(np.float32)

    np.savez(
        os.path.join(opt["_root"], "impulse_response.npz"),
        hists=hists,
        Rts=Rts,
        power=opt["light"]["power"],
        fov=min(opt["light"]["fov"], opt["camera"]["fov"]),
        bin_size=opt["camera"]["bin_size"],
        num_bins=opt["camera"]["num_bins"],
        camera_loc=loc_save,
        camera_normal=normal_save
    )
    print("Simulation finished!")


def simulate_sensor_response(opt):
    # load impulse response
    assert os.path.exists(opt["data_path"])
    data = np.load(opt["data_path"])
    hists = data["hists"]
    hists_impulse = np.copy(hists)
    Rts, power, fov = data["Rts"], data.get("power", 1), data["fov"]
    bin_size, num_bins = data["bin_size"], data["num_bins"]

    print("Simulation started...")

    # convolve impulse response with laser pulse
    # pulse has Gaussian shape and unit peak power
    # width (Gaussian stddev) given by full-width at half-maximum (FWHM)
    # FWHM given in picoseconds (e.g., 90ps)
    pulse_kernel = Light.pulse_kernel(opt["fwhm"])
    tics = np.linspace(0, num_bins, num_bins + 1)[1:-1]
    tics = (tics - 0.5 * num_bins) * bin_size / 3e8 * 1e12  # unit: ps
    pulse = pulse_kernel(tics)
    for i in range(len(hists)):
        hists[i] = np.convolve(hists[i], pulse, mode="same")
    hists_pulse_conv = np.copy(hists)

    # scale histogram and add noise floor
    # H' = gamma * (s * H + n), H is impulse response
    # s given by laser power, n given by noise
    # gamma given by quantum efficiency, lens aperture and pixel size
    scale = opt["power"] / power
    hists = scale * hists + opt["noise"]
    hists_scale_noise = np.copy(hists)

    # simulate pile-up distortion
    # H' gives per-bin Poisson rates ri
    # c.f. Equation (3) in Gupta et al., ICCV 19
    r_sum = np.cumsum(np.concatenate((np.zeros((len(hists), 1)), hists[:, :-1]), axis=1), axis=1)
    p = (1 - np.exp(-hists)) * np.exp(-r_sum)
    p = np.concatenate((p, np.clip(1 - p.sum(axis=1, keepdims=True), 0, 1)), axis=1)

    # sample photons
    # simulate N laser cycles
    # histogram values controlled by multinomial distribution
    # c.f. Equation (4) in Gupta et al., ICCV 19
    # hit rate is the percentage of cycles with detected photons
    # low-light regime (negligible pile-up effect) has low hit rate (1-5%)
    hit_rates = []
    for i in range(len(hists)):
        counts = np.random.multinomial(opt["num_cycles"], p[i] / p[i].sum())
        hists[i] = counts[:-1]  # last bin counts misses
        hit_rates.append(1 - counts[-1] / opt["num_cycles"])
    hit_rate = np.array(hit_rates).mean()
    hists_pileup = np.copy(hists)

    # load time jitter
    # jitter is experimentally measured, given by Hernandez et al. TCI 2017
    # normalize jitter counts to obtain jitter PDF
    assert os.path.exists(opt["jitter_path"])
    jitter = np.load(opt["jitter_path"])
    jitter_pdf = jitter["counts"] / jitter["counts"].sum()
    jitter_size = jitter["t"] * 3e8  # unit: m

    # apply time jitter
    # sample per-photon jitter size according to jitter PDF
    # add jitter and keep photons within histogram bin range
    tics = np.linspace(0, num_bins, num_bins + 1) * bin_size
    hists = hists.astype(int)
    for i in range(len(hists)):
        num_photons = int(hists[i].sum())
        t = np.clip(
            tics[:-1].repeat(hists[i])
            + np.random.rand(num_photons) * bin_size
            + np.random.choice(jitter_size, num_photons, p=jitter_pdf),
            bin_size / 2,
            tics[-1] - bin_size / 2,
        )
        hists[i] = np.histogram(t, bins=tics)[0]

    # convert jitter into convolution kernel
    jitter_kernel = _make_jitter_kernel(num_bins, bin_size, jitter_pdf, jitter_size)

    # save sensor response
    np.savez(
        os.path.join(opt["_root"], "sensor_response.npz"),
        hists_impulse=hists_impulse,
        hists_pulse_conv=hists_pulse_conv,
        hists_scale_noise=hists_scale_noise,
        hists_pileup=hists_pileup,
        hists_jitter=hists,
        Rts=Rts,
        fov=fov,
        bin_size=bin_size,
        num_bins=num_bins,
        fwhm=opt["fwhm"],
        power=opt["power"],
        noise=opt["noise"],
        num_cycles=opt["num_cycles"],
        hit_rate=hit_rate,
        pulse_kernel=pulse,
        jitter_kernel=jitter_kernel,
    )
    print("Simulation finished!")


def _make_jitter_kernel(num_bins, bin_size, jitter_pdf, jitter_size):
    """Convert time jitter into convolution kernel."""

    tics = np.linspace(-0.5 * num_bins, 0.5 * num_bins, num_bins + 1) * bin_size
    kernel = np.zeros(num_bins)
    src_idx = dst_idx = 0
    while tics[dst_idx] < jitter_size[0]:
        dst_idx += 1
    curr_t, cum_sum = jitter_size[0], jitter_pdf[0]
    while src_idx < len(jitter_size) - 1 and dst_idx < len(tics) - 1:
        if tics[dst_idx] < jitter_size[src_idx]:
            if tics[dst_idx + 1] < jitter_size[src_idx]:
                cum_sum += jitter_pdf[src_idx] * (
                    (tics[dst_idx + 1] - tics[dst_idx]) / (jitter_size[src_idx] - jitter_size[src_idx - 1])
                )
                curr_t = tics[dst_idx + 1]
                kernel[dst_idx] = cum_sum
            else:
                cum_sum += jitter_pdf[src_idx] * (
                    (jitter_size[src_idx] - curr_t) / (jitter_size[src_idx] - jitter_size[src_idx - 1])
                )
                curr_t = jitter_size[src_idx]
            dst_idx += 1
        else:
            if jitter_size[src_idx + 1] < tics[dst_idx]:
                cum_sum += jitter_pdf[src_idx + 1]
                curr_t = jitter_size[src_idx + 1]
            else:
                cum_sum += jitter_pdf[src_idx + 1] * (
                    (tics[dst_idx] - curr_t) / (jitter_size[src_idx + 1] - jitter_size[src_idx])
                )
                curr_t = tics[dst_idx]
                kernel[dst_idx] = cum_sum
            src_idx += 1
    while dst_idx < len(kernel):
        kernel[dst_idx] = cum_sum
        dst_idx += 1
    kernel = kernel[1:] - kernel[:-1]
    return kernel
