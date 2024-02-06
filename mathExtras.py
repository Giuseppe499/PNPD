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

import numpy as np
from numpy.fft import fft2, ifft2


def sInner(x: np.array, y: np.array = None):
    if y is not None:
        return np.inner(x.ravel(), y.ravel())
    return np.inner(x.ravel(), x.ravel())


def softTreshold(alpha, x: np.array):
    return np.sign(x) * np.maximum(abs(x) - alpha, 0)


def gradLeastSquares(x, bFFT, psfFFT, psfFFTC):
    xFFT = fft2(x)
    return np.real(ifft2(psfFFTC * (psfFFT * xFFT - bFFT)))

def gradLeastSquaresRGB(x, bFFT, psfFFT, psfFFTC):
    grad = np.zeros(x.shape)
    for i in range(3):
        grad[:,:,i] = gradLeastSquares(x[:,:,i], bFFT[:,:,i], psfFFT, psfFFTC)
    return grad
    

def mulPInLeastSquares(mu, x, psfAbsSq):
    return np.real(ifft2(fft2(x) / (psfAbsSq + mu)))

def mulPInLeastSquaresRGB(mu, x, psfAbsSq):
    mul = np.zeros(x.shape)
    for i in range(3):
        mul[:,:,i] = mulPInLeastSquares(mu, x[:,:,i], psfAbsSq)
    return mul

def mulPLeastSquares(mu, x, psfAbsSq):
    return np.real(ifft2(fft2(x) * (psfAbsSq + mu)))

def grad2D(m: np.array):
    dx = np.roll(m, -1, axis=-2) - m
    dy = np.roll(m, -1, axis=-1) - m
    # Comment for periodic boundary conditions
    dx[...,-1, :] = 0
    dy[...,:, -1] = 0
    return np.stack((dx, dy))

def grad2Drgb(m: np.array):
    return np.stack([grad2D(m[:,:,i]) for i in range(3)], axis=-1)

def div2D(dxdy: np.array):
    dx = dxdy[0, ...]
    dy = dxdy[1, ...]
    fx = np.roll(dx, 1, axis=-2) - dx
    fy = np.roll(dy, 1, axis=-1) - dy
    fx[..., 0, :] = -dx[..., 0, :]
    fx[..., -1, :] = dx[..., -2, :]
    fy[..., :, 0] = -dy[..., :, 0]
    fy[..., :, -1] = dy[..., :, -2]
    return fx + fy

def div2Drgb(dxdy: np.array):
    return np.stack([div2D(dxdy[:,:,:,i]) for i in range(3)], axis=-1)

def proxhsTV(lam: float, dxdy: np.array):
    dx = dxdy[0,...]
    dy = dxdy[1,...]
    factor = np.sqrt(dx*dx + dy*dy)
    factor = np.maximum(factor/lam, 1)
    factor = np.stack((factor, factor))
    return dxdy / factor

def proxhsTVrgb(lam: float, dxdy: np.array):
    for i in range(3):
        dxdy[:,:,:,i] = proxhsTV(lam, dxdy[:,:,:,i])
    return dxdy


def fftConvolve2D(in1, in2):
    return np.real(ifft2(fft2(in1) * fft2(in2)))

def fftConvolve2Drgb(in1, in2):
    return np.stack([fftConvolve2D(in1[:,:,i], in2) for i in range(3)], axis=-1)


def generatePsfMatrix(size: int, sigma: float) -> np.array:
    """
    Generate a Point Spread Function (PSF) matrix.

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

def centerCrop(image, target_size=(256, 256)):
    h = image.shape[0]
    w = image.shape[1]
    th, tw = target_size

    # Calculate the starting point for the crop
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    # Perform the crop
    cropped_image = image[i:i+th, j:j+tw]

    return cropped_image

def padWithZeros(image, target_size=(256, 256)):
    h, w, _ = image.shape
    th, tw = target_size

    # Calculate the starting point for the crop
    i = int(round((th - h) / 2.))
    j = int(round((tw - w) / 2.))

    # Perform the crop
    padded_image = np.zeros((th, tw, 3))
    padded_image[i:i+h, j:j+w] = image

    return padded_image
