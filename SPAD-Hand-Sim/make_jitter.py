import os, json
import matplotlib.pyplot as plt
import numpy as np


num_bins = 128
bin_size = 0.2788732647895813


def make_jitter(data_path):
    for json_file in os.listdir(data_path):
        hists = json.load(open(data_path + json_file))
        jitters = []
        for i in range(8):
            jitter = np.array(hists[str(i + 1)]["hists"][0][0]).astype(np.float64)
            print(jitter.sum())
            jitter /= jitter.sum()
            jitters.append(jitter)
            plt.plot(jitter, label=data_path)

            # response = np.zeros_like(jitter)
            # for j in range(9):
            #     zone = np.array(hists[str(i)]["hists"][0][j + 1])
            #     print(zone.sum())
            #     response += zone
            # plt.plot(response)
        break

    jitter_kernels = []
    offset = 9.523497581481934 / bin_size
    for i in range(len(jitters)):
        jitter = jitters[i]
        kernel = np.zeros(num_bins)
        src_idx = dst_idx = 0
        tics = np.linspace(-0.5 * num_bins, 0.5 * num_bins, num_bins + 1) / bin_size
        while tics[dst_idx] < offset:
            dst_idx += 1
        curr_t, cum_sum = offset, jitter[0]
        while src_idx < 127 and dst_idx < len(tics) - 1:
            if tics[dst_idx] < src_idx + offset:
                if tics[dst_idx + 1] < src_idx + offset:
                    cum_sum += jitter[src_idx] * (tics[dst_idx + 1] - tics[dst_idx])
                    curr_t = tics[dst_idx + 1]
                    kernel[dst_idx] = cum_sum
                else:
                    cum_sum += jitter[src_idx] * (src_idx + offset - curr_t)
                    curr_t = src_idx + offset
                dst_idx += 1
            else:
                if src_idx + 1 + offset < tics[dst_idx]:
                    cum_sum += jitter[src_idx + 1]
                    curr_t = src_idx + 1 + offset
                else:
                    cum_sum += jitter[src_idx + 1] * (tics[dst_idx] - curr_t)
                    curr_t = tics[dst_idx]
                    kernel[dst_idx] = cum_sum
                src_idx += 1
        while dst_idx < len(kernel):
            kernel[dst_idx] = cum_sum
            dst_idx += 1
        kernel = kernel[1:] - kernel[:-1]
        jitter_kernels.append(kernel)

    jitter_kernels = np.vstack(jitter_kernels).reshape(8, 1, 127)
    np.savez(
        "data/new_jitter",
        jitter_pdf=np.vstack(jitters),
        jitter_kernel=jitter_kernels,
    )


# make_jitter("data/real/2024-09-20_test_data/tof/")
# make_jitter("data/real/2024-10-03_calibration/tof/")
make_jitter("jitter-path")
