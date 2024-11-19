import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import SPADHistNV

root = "data/real/2024-10-08_planes/one_sensor/"
label = json.load(open(root + "labels.json"))

spad_layer = SPADHistNV(
    num_cameras=1,
    num_bins=128,
    resolution=256,
    fov=33,
    cube_mode=True,
    single_camera=True,
)
hand_transl = torch.tensor([[-1.02, -0.5, -0.5]]).cuda()
global_rot = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().cuda()
peaks_x = []
peaks_y = []

dist = 9
for i in range(dist, dist + 1):
    hand_transl[0, 0] = -1 - i * 0.01
    hist = (
        spad_layer(
            None,
            None,
            hand_transl,
            global_rot,
            image_path="results/single_camera.png",
            random_sample=False,
        )
        .detach()
        .cpu()
        .numpy()
    )
    plt.plot(hist[0], label=str(i))
    peaks_x.append(hist[0].argmax())
    peaks_y.append(hist[0].max())
    # plt.legend()
print(peaks_x, peaks_y)
plt.plot(peaks_x, peaks_y)

for filename in label:
    distance = label[filename]["object"]
    hists = json.load(open(root + filename + ".json"))
    if distance != str(dist) + "cm":
        continue
    for hist in hists[1:]:
        response = np.array(hist["hists"]).astype(np.float64)
        response = np.sum(response, axis=0)  # combine 9 sub-cameras
        print(response.sum(), response.argmax(), response[:15].sum())
        response /= spad_layer.num_cycles
        # response -= np.min(response, axis=0)
        # plt.scatter(
        #     response.argmax(), response.max(), label=distance
        # )
        plt.plot(response)
        plt.legend()

plt.show()
