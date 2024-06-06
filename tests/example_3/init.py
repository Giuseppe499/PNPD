from PIL import Image
import numpy as np
from math_extras import generate_gaussian_PSF, move_psf_center
from tests.generate_blurred_image import generate_blurred_image
from tests.constants import *

IMAGE_PATH = "../images/cameraman.tif"
DATA_SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/deblur_problem_data"
EXAMPLE_NAME = "example_3"

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

    blurred = generate_blurred_image(
        image, noisePercent=0.05, psf=psf, save_path=DATA_SAVE_PATH
    ).blurred

    from matplotlib import pyplot as plt
    from tests.plot_extras import plot_image_psf_blurred
    plot_image_psf_blurred(image, psf_centered, blurred)
    plt.show()