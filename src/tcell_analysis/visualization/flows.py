import matplotlib.pyplot as plt
import numpy as np


def save_flow_quiver_plot(rgb_flow, flow_yx, cellprob, out_path):

    dy, dx = flow_yx[0], flow_yx[1]
    step = 10
    Y, X = np.mgrid[0 : dy.shape[0] : step, 0 : dy.shape[1] : step]
    dy_sub = dy[::step, ::step]
    dx_sub = dx[::step, ::step]

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.title("Flow RGB Image")
    plt.imshow(rgb_flow)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Vector Field (Quiver)")
    plt.imshow(cellprob, cmap="gray")
    plt.quiver(X, Y, dx_sub, dy_sub, scale=50, color="cyan", width=0.002)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Cell Probability")
    plt.imshow(cellprob, cmap="hot")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
