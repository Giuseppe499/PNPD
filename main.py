from PIL import Image
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

IMGPATH = "cameraman.tif"

def generatePsfMatrix(size: int, sigma: float) -> np.array:
    """
    Generate a Point Spread Function (PSF) matrix.

    Parameters:
    - size: Size of the PSF matrix (e.g., size=11 for an 11x11 matrix)
    - sigma: Standard deviation of the Gaussian PSF

    Returns:
    - psfMatrix: PSF matrix
    """
    # Create a grid of coordinates centered at the PSF matrix
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    xx, yy = np.meshgrid(x, y)

    # Calculate the 2D Gaussian PSF
    psfMatrix = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    # Normalize the PSF matrix to sum to 1
    psfMatrix /= np.sum(psfMatrix)

    return psfMatrix


if __name__ == '__main__':
    image = Image.open(IMGPATH)
    image = np.asarray(image) / 255
    print(image.shape)

    psf = generatePsfMatrix(256, 16)
    noise = np.random.normal(size = image.shape, scale = 1e-2)
    conv = sp.signal.convolve(image, psf, "same")

    b = conv + noise

    #plt.imshow(psf)

    f, axs = plt.subplots(1, 3)
    axs[0].imshow(conv, cmap="gray", vmin=0, vmax=1)
    axs[1].imshow(noise, cmap="gray")
    axs[2].imshow(b, cmap="gray", vmin=0, vmax=1)

    f, axs = plt.subplots(1,2)
    axs[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axs[1].imshow(b, cmap="gray", vmin=0, vmax=1)

    plt.show()
