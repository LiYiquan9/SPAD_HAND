import yaml


DEFAULTS = {
    "seed": 1234567891,

    "sdf": {
        "hid_dim": 256,
        "num_freqs": 6,
        "num_layers": 8,
        "skips": (4, ),
        "weight_norm": True,
        "geometric_init": True,
        "r_init": 0.25,
    },

    "deviation": {
        "init_val": 0.3,
    },

    "brdf": {
        "init_val": 0.0,
    },

    "render": {
        "num_ray_samples": ((64, 8), 64),
        "num_step_samples": (256, 32),
        "radius": 1.0,
    },

    "batch_size": 4,

    "lr": 5.e-4,
    "min_lr": 2.5e-5,

    "eikonal_weight": 0.1,
    "entropy_weight": 0.001,

    "itrs": 300000,
    "warmup_itrs": 5000,
    "cos_anneal_itrs": 50000,

    "log_interval": 100,
    "plot_interval": 1000,
    "checkpoint_interval": 10000,
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_opt(filepath):
    with open(filepath, "r") as fd:
        opt = yaml.load(fd, Loader=yaml.FullLoader)
    
    _merge(DEFAULTS, opt)
    return opt