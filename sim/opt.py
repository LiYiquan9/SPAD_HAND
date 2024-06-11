import yaml


IMPULSE_DEFAULTS = {
    "setup_method": "fibonacci",
    "setup": {
        "min_elevation": 30,
        "num_cameras": 256,
        "radius": 1,
    },

    "light": {
        "fov": 60,
        "power": 1,
        "is_area": False,
    },

    "camera": {
        "fov": 60,
        "bin_size": 0.02,
        "num_bins": 128,
        "is_area": True,
    },

    "render_method": "path",
    "render": {
        "albedo": 0.5,
        "num_samples": 50000,
        "normal_mode": "vn",
    },
}


SENSOR_DEFAULTS = {
    "fwhm": 90,
    "power": 100,
    "noise": 0.001,
    "num_cycles": 4e6,
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_opt(filepath, mode="impulse"):
    with open(filepath, "r") as fd:
        opt = yaml.load(fd, Loader=yaml.FullLoader)
    
    _merge(IMPULSE_DEFAULTS if mode == "impulse" else SENSOR_DEFAULTS, opt)
    return opt