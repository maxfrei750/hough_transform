import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from hough_transform import hough_line, rgb2gray


def plot_hough_transform():
    image_path = "lines.png"
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

    ax = plt.gca()

    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlim([-90, 90])

    ax.set_xlabel("Angle/°")
    ax.set_ylabel("Directed distance from origin/px")

    plt.savefig("accumulator.pdf")
    plt.show()


if __name__ == "__main__":
    plot_hough_transform()
