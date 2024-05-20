from PIL import Image
import numpy as np
from math_extras import move_psf_center
import scipy.io as sio
from tests.generate_blurred_image import generate_blurred_image
from tests.constants import *

IMAGE_PATH = "../images/peppers.tiff"
DATA_SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/deblur_problem_data"
EXAMPLE_NAME = "example_2"

if __name__ == "__main__":

    image = Image.open(IMAGE_PATH).convert('L')
    image = np.asarray(image) / 255
    image = image[::2, ::2]

    n = image.shape[0]
    print(f"Image size: {image.shape}")

    # Generate PSF
    psf_centered = sio.loadmat('kernel.mat')['motion8']
    m = psf_centered.shape[0]
    psf_centered = np.pad(psf_centered, ((n-m)//2 + 1, (n-m)//2) , 'constant')

    # Move PSF center to the top-left corner
    psf = move_psf_center(psf_centered)

    blurred = generate_blurred_image(
        image, noisePercent=0.01, psf=psf, save_path=DATA_SAVE_PATH
    ).blurred

    from matplotlib import pyplot as plt
    from tests.plot_extras import plot_image_psf_blurred
    plot_image_psf_blurred(image, psf_centered, blurred)
    plt.show()