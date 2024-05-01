"""
PNPD implementation

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

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.polynomial import Polynomial


def scalar_product(x: np.array, y: np.array = None):
    if y is not None:
        return np.inner(x.ravel(), y.ravel())
    return np.inner(x.ravel(), x.ravel())


def soft_threshold(alpha, x: np.array):
    return np.sign(x) * np.maximum(abs(x) - alpha, 0)


def gradient_convolution_least_squares(x, bFFT, psfFFT, psfFFTC, axes = (0, 1)):
    xFFT = fft2(x, axes=axes)
    return np.real(ifft2(psfFFTC * (psfFFT * xFFT - bFFT), axes=axes))    

def multiply_P_inverse(p: Polynomial, x, psfAbsSq, axes = (0, 1)):
    return np.real(ifft2(fft2(x, axes=axes) / p(psfAbsSq), axes=axes))

def multiply_P(p: Polynomial, x, psfAbsSq, axes = (0, 1)):
    return np.real(ifft2(fft2(x, axes=axes) * p(psfAbsSq), axes=axes))

def gradient_2D_signal(m: np.array):
    dx = np.roll(m, -1, axis=-2) - m
    dy = np.roll(m, -1, axis=-1) - m
    # Comment for periodic boundary conditions
    dx[...,-1, :] = 0
    dy[...,:, -1] = 0
    return np.stack((dx, dy))

def divergence_2D_signal(dxdy: np.array):
    dx = dxdy[0, ...]
    dy = dxdy[1, ...]
    fx = np.roll(dx, 1, axis=-2) - dx
    fy = np.roll(dy, 1, axis=-1) - dy
    fx[..., 0, :] = -dx[..., 0, :]
    fx[..., -1, :] = dx[..., -2, :]
    fy[..., :, 0] = -dy[..., :, 0]
    fy[..., :, -1] = dy[..., :, -2]
    return fx + fy

def prox_h_star_TV(lam: float, dxdy: np.array):
    dx = dxdy[0,...]
    dy = dxdy[1,...]
    factor = np.sqrt(dx*dx + dy*dy)
    factor = np.maximum(factor/lam, 1)
    factor = np.stack((factor, factor))
    return dxdy / factor

def convolve_2D_fft(in1, in2, axes = (0, 1)):
    return np.real(ifft2(fft2(in1, axes=axes) * fft2(in2, axes=axes), axes=axes))

def generate_gaussian_PSF(size: int, sigma: float) -> np.array:
    """
    Generate a Gaussian Point Spread Function (PSF) matrix.

    Parameters:
    - size: Size of the PSF matrix (e.g., size=11 for a 11x11 matrix)
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

def generate_out_of_focus_PSF(size: int, radius: int) -> np.array:
    """
    Generate a PSF matrix for an out-of-focus lens.

    Parameters:
    - size: Size of the PSF matrix (e.g., size=11 for a 11x11 matrix)
    - radius: Radius of the out-of-focus lens

    Returns:
    - psfMatrix: PSF matrix
    """
    # Create a grid of coordinates centered at the PSF matrix
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    xx, yy = np.meshgrid(x, y)

    # Calculate the 2D PSF
    psfMatrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if xx[i, j] ** 2 + yy[i, j] ** 2 <= radius ** 2:
                psfMatrix[i, j] = 1

    # Normalize the PSF matrix to sum to 1
    psfMatrix /= np.sum(psfMatrix)

    return psfMatrix

def center_crop(image, target_size: tuple[int, int]):
    h = image.shape[0]
    w = image.shape[1]
    th, tw = target_size

    # Calculate the starting point for the crop
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    # Perform the crop
    cropped_image = image[i:i+th, j:j+tw]

    return cropped_image

def pad_with_zeros(image: np.ndarray, target_size: tuple[int, int]):
    h, w, _ = image.shape
    th, tw = target_size

    i = int(round((th - h) / 2.))
    j = int(round((tw - w) / 2.))

    padded_image = np.zeros((th, tw, 3))
    padded_image[i:i+h, j:j+w] = image

    return padded_image