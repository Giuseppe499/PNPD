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

from context import PNPD

from PIL import Image
import numpy as np
from numpy.fft import fft2
from PNPD.math_extras import convolve_2D_fft, generate_gaussian_PSF, generate_out_of_focus_PSF, scalar_product
from dataclasses import dataclass
from PNPD.utilities import save_data

@dataclass
class DeblurProblemData:
    image: np.ndarray = None
    convolved: np.ndarray = None
    noise: np.ndarray = None
    blurred: np.ndarray = None
    bFFT: np.ndarray = None
    psf: np.ndarray = None
    psfFFT: np.ndarray = None
    psfFFTC: np.ndarray = None
    psfAbsSq: np.ndarray = None
    image: np.ndarray = None
    noiseNormSqd: float = None

def generate_blurred_image(image, noisePercent, psf, save_path=None) -> DeblurProblemData:
    data = DeblurProblemData()
    data.image = image
    data.psf = psf

    # Generate blurred image
    data.convolved = convolve_2D_fft(image, psf)

    # Generate noise
    data.noise = np.random.normal(size=image.shape)
    data.noise *= noisePercent * np.linalg.norm(data.convolved) / np.linalg.norm(data.noise)
    data.noiseNormSqd = scalar_product(data.noise.ravel())

    # Add noise to blurred image
    data.blurred = np.clip(data.convolved + data.noise, 0, 1)

    # Generate FFTs
    data.bFFT = fft2(data.blurred)
    data.psfFFT = fft2(data.psf)
    data.psfFFTC = np.conjugate(data.psfFFT)
    data.psfAbsSq = data.psfFFTC * data.psfFFT

    # Save data
    if save_path is not None:
        save_data(save_path, data)

    return data

