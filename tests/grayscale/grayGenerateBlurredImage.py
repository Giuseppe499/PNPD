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

from PIL import Image
import numpy as np
from numpy.fft import fft2
from mathExtras import fftConvolve2D, generatePsfMatrix, sInner
import grayConfig
import os

def main():
    image = grayConfig.image
    noisePercent = grayConfig.noisePercent
    psf = grayConfig.psf
    psfBT = grayConfig.psfBT

    # Generate blurred image
    conv = fftConvolve2D(image, psf)

    # Generate noise
    noise = np.random.normal(size=image.shape)
    noise *= noisePercent * np.linalg.norm(conv) / np.linalg.norm(noise)
    noiseNormSqd = sInner(noise.ravel())

    # Add noise to blurred image
    b = np.clip(conv + noise, 0, 1)

    # Generate FFTs
    bFFT = fft2(b)
    psfFFT = fft2(psf)
    psfFFTC = np.conjugate(psfFFT)
    psfAbsSq = psfFFTC * psfFFT

    # Save data
    filename = f'./npz/{grayConfig.prefix}/Blurred.npz'    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, conv=conv, noise=noise, b=b, bFFT=bFFT, psf=psf, psfBT=psfBT, psfFFT=psfFFT, psfFFTC=psfFFTC, psfAbsSq=psfAbsSq, image=image, noiseNormSqd=noiseNormSqd)


    if __name__ == "__main__":
        IMGPATH = "cameraman.tif"
        grayConfig.noisePercent = 0.02

        image = Image.open(IMGPATH)
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        grayConfig.image = image
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = generatePsfMatrix(n, 1.6)
        psfBT = psf.copy()
        # Center PSF
        psf = np.roll(psf, (-psf.shape[0] // 2, -psf.shape[0] // 2), axis=(0, 1))
        grayConfig.psf = psf
        grayConfig.psfBT = psfBT
        main()
