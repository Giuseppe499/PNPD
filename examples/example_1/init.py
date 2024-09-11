from context import PNPD, examples

from PIL import Image
import numpy as np
from PNPD.math_extras import generate_gaussian_PSF, move_psf_center
from examples.generate_blurred_image import generate_blurred_image
from examples.constants import *

IMAGE_PATH = "../images/cameraman.tif"
DATA_SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/deblur_problem_data"
EXAMPLE_NAME = "example_1"

if __name__ == "__main__":

    image = Image.open(IMAGE_PATH)
    image = np.asarray(image) / 255
    image = image[::2, ::2]

    n = image.shape[0]
    print(f"Image size: {image.shape}")

    # Generate PSF
    psf_centered = generate_gaussian_PSF(n, 2)

    # Move PSF center to the top-left corner
    psf = move_psf_center(psf_centered)

    data = generate_blurred_image(
        image, noisePercent=0.01, psf=psf, save_path=DATA_SAVE_PATH
    )

    from matplotlib import pyplot as plt
    from PNPD.plot_extras import plot_images
    from PNPD.math_extras import center_crop
    import os
    psf_center_crop = center_crop(psf_centered, (20,20)) / psf_centered.max()
    plot_images(
        [image, psf_center_crop, data.convolved, data.blurred],
        ["$x$\n(ground truth)",
         "PSF\n(center crop of size $20\\times20$)", 
         "$\\tilde b = x\circledast_{\\text{2D}}$ PSF",
         "$b = \\tilde b + e$"],
         (2,2))
    save_path = ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + "problem.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    save_path = ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for im, name in zip((image, psf_center_crop, data.blurred),
                         ("ground_truth", "psf", "observed")):
        plot_images([im])
        plt.savefig(save_path + f"problem_{name}.pdf")

    plt.show()