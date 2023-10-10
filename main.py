"""
NPD implementation

Copyright (C) 2023 Giuseppe Scarlato

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Giuseppe Scarlato"
__contact__ = "giuseppe499[at]live.it"
__copyright__ = "Copyright 2023, Giuseppe Scarlato"
__date__ = "2023/10/06"
__license__ = "GPLv3"

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

def fftConvolve2D(in1, in2):
    return np.real(np.fft.ifft2(np.fft.fft2(in1) * np.fft.fft2(in2)))

def deConvolve2D(conv, psf, epsilon: float):
    return np.real(np.fft.ifft2(np.fft.fft2(conv) / np.clip(np.fft.fft2(psf),epsilon, None) ))

def deConvolve2DThikonov(conv, psf, alpha):
    psfFFT = np.fft.fft2(psf)
    psfFFTC = np.conjugate(psfFFT)
    convFFT = np.fft.fft2(conv)
    return np.real(np.fft.ifft2( psfFFTC*convFFT/(psfFFTC*psfFFT + alpha) ) )

if __name__ == '__main__':
    # Load image
    image = Image.open(IMGPATH)
    image = np.asarray(image) / 255
    n = image.shape[0];
    print(image.shape)

    # Generate PSF
    psf = generatePsfMatrix(n, 8)
    # Center PSF
    psf = np.roll(psf, (-psf.shape[0] // 2, -psf.shape[0] // 2), axis=(0, 1))
    # Generate noise
    noise = np.random.normal(size = image.shape, scale = 1e-4)
    # Generate blurred image
    conv = fftConvolve2D(image, psf)

    # Add noise to blurred image
    b = np.clip(conv + noise,0,1)
    imRec = deConvolve2D(b, psf, 1e-3)
    imRecThikonov = deConvolve2DThikonov(b, psf, 1e-6)

    # Show PSF
    #plt.imshow(psf)

    # Show blurred image, noise and the sum of these
    f, axs = plt.subplots(1, 3)
    axs[0].imshow(conv, cmap="gray", vmin=0, vmax=1)
    axs[1].imshow(noise, cmap="gray")
    axs[2].imshow(b, cmap="gray", vmin=0, vmax=1)

    # Show original image and blurred image
    f, axs = plt.subplots(1,2)
    axs[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axs[1].imshow(b, cmap="gray", vmin=0, vmax=1)
    f, axs = plt.subplots(1, 3)
    axs[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axs[1].imshow(imRec, cmap="gray", vmin=0, vmax=1)
    axs[2].imshow(imRecThikonov, cmap="gray", vmin=0, vmax=1)

    plt.show()
