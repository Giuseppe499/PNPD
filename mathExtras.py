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


def sInner(x: np.array):
    return np.inner(x, x)


def softTreshold(alpha, x: np.array):
    return np.sign(x) * np.maximum(abs(x) - alpha, 0)


def gradLeastSquares(x, bFFT, psfFFT, psfFFTC):
    xFFT = fft2(x)
    return np.real(ifft2(psfFFTC * (psfFFT * xFFT - bFFT)))


def grad2D(m: np.array):
    dx = np.roll(m, -1, axis=0) - m
    dy = np.roll(m, -1, axis=1) - m
    # Comment for periodic boundary conditions
    dx[-1, :] = 0
    dy[:, -1] = 0

    return np.stack((dx, dy))


def div2D(dxdy: np.array):
    dx = dxdy[0, :, :]
    dy = dxdy[1, :, :]
    fx = np.roll(dx, 1, axis=0) - dx
    fy = np.roll(dy, 1, axis=1) - dy
    fx[0, :] = -dx[0, :]
    fx[-1, :] = dx[-2, :]
    fy[:, 0] = -dy[:, 0]
    fy[:, -1] = dy[:, -2]
    return fx + fy

def proxhsTV(lam: float, dxdy: np.array):
    dx = dxdy[0, :, :]
    dy = dxdy[1, :, :]
    factor = np.sqrt(dx*dx + dy*dy)
    factor = np.maximum(factor/lam, 1)
    factor = np.stack((factor, factor))
    return dxdy / factor


def fftConvolve2D(in1, in2):
    return np.real(ifft2(fft2(in1) * fft2(in2)))


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
