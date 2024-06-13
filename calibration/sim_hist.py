import torch
import torch.nn.functional as F
import numpy as np


from calibration.render import evaluate_path_integral, evaluate_angular_integral
from calibration.structs import Anchor, Light, Camera  # to .structs
from sim import load_opt


# @torch.jit.script
def convolve(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv1d(
        x.unsqueeze(0), kernel,
        padding=x.shape[-1] // 2 - 1, groups=kernel.shape[0]
    )[0]


# @torch.jit.script
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

def sim_hist(vertices, mesh, cam_loc, cam_norm):
    
    impulse_response = sim_impulse_response(vertices, mesh, cam_loc, cam_norm)
    
    return impulse_response['hists']
    
    ### the following part is for differentiable sensor simulation ###
    
    # data = np.load("calibration/data/sensor_response.npz")
    # pulse_kernel = data["pulse_kernel"].astype(np.float32)
    # pulse_kernel = torch.from_numpy(pulse_kernel)
    # pulse_kernel = pulse_kernel.repeat(1, 1).unsqueeze(1)
    # pulse_kernel = torch.flip(pulse_kernel, dims=(-1, ))
    
    # jitter_kernel = data["jitter_kernel"].astype(np.float32)
    # jitter_kernel = torch.from_numpy(jitter_kernel)
    # jitter_kernel = jitter_kernel.repeat(1, 1).unsqueeze(1)
    # jitter_kernel = torch.flip(jitter_kernel, dims=(-1, ))

    # power = float(data["power"])
    # noise = float(data["noise"])
    
    # pred_hists = run_sensor_model(
    #         hists=impulse_response["hists"].to(torch.float32), # torch.Size([1, 64])
    #         pulse_kernel=pulse_kernel,
    #         jitter_kernel=jitter_kernel,
    #         power=power,
    #         noise=noise,
    #     )
    
    # return pred_hists
    

    
def sim_impulse_response(vertices, mesh, cam_loc, cam_norm):
    
    opt = load_opt("opts/simulate/impulse_response/template.yaml")
    
    locs = cam_loc
    
    normals = cam_norm
    lights, cameras = [], []
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

    # select rendering method
    # path integral: sample points on object surface
    # angular integral: sample rays from within light / camera FoV
    render_fn = evaluate_path_integral if opt["render_method"] == "path" else evaluate_angular_integral

    # run simulation
    hists, Rts = [], []
    for idx, (light, camera) in enumerate(zip(lights, cameras)):
      
        hist = render_fn(vertices, mesh, light, camera, **opt["render"]) # TODO: make render function differentiable in pytorch
        
        hists.append(hist)
        # Rts.append(camera.a2w)
        print(f"\t{(idx + 1):0{len(str(len(cameras)))}d} done.")

    hists = torch.stack(hists)
    Rts = np.array(Rts).astype(np.float32)


    sim_impulse_data = {}
    sim_impulse_data["hists"] = hists
    sim_impulse_data["Rts"] = Rts
    sim_impulse_data["power"] = opt["light"]["power"]
    sim_impulse_data["fov"] = min(opt["light"]["fov"], opt["camera"]["fov"])
    sim_impulse_data["bin_size"] = opt["camera"]["bin_size"]
    sim_impulse_data["num_bins"] = opt["camera"]["num_bins"]
    
    return sim_impulse_data
    
