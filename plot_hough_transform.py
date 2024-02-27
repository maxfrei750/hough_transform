import pathlib

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pdfCropMargins
from hough_transform import hough_line, rgb2gray

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parent


def plot_hough_transform():
    image_path = SCRIPT_ROOT / "lines.png"
    image = imageio.imread(image_path)
    if image.ndim == 3:
        image = rgb2gray(image)

    accumulator, angles, distances = hough_line(
        image, angle_step=0.1, num_length_step_factor=32
    )

    plt.imshow(
        accumulator,
        cmap="viridis",
        aspect="auto",
        extent=[
            np.rad2deg(angles[-1]),
            np.rad2deg(angles[0]),
            distances[-1],
            distances[0],
        ],
    )

    plt.colorbar(label="Votes")

    ax = plt.gca()

    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlim([-90, 90])

    ax.set_xlabel(r"Angle $\varphi$ of distance vector/°")
    ax.set_ylabel(r"Length $|\vec{p}|$ of distance vector/px")

    output_path = SCRIPT_ROOT / "accumulator.pdf"
    plt.savefig(output_path)

    pdfCropMargins.crop(
        [
            "-p",
            "10",
            "--modifyOriginal",
            str(output_path),
        ]
    )

    # plt.show()
    plt.close()


if __name__ == "__main__":
    plot_hough_transform()
