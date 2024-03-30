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
from mathExtras import fftConvolve2D, gaussianPSF, outOfFocusPSF, sInner
import os

def generateBlurredImage(image, noisePercent, psf, savePath = None):
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
    if savePath is not None:
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        np.savez(savePath, conv=conv, noise=noise, b=b, bFFT=bFFT, psf=psf, psfFFT=psfFFT, psfFFTC=psfFFTC, psfAbsSq=psfAbsSq, image=image, noiseNormSqd=noiseNormSqd)

    return conv, noise, b, bFFT, psf, psfFFT, psfFFTC, psfAbsSq, image, noiseNormSqd


RGB = True

if __name__ == "__main__":

    if not RGB:
        IMGPATH = "cameraman.tif"
    else:
        IMGPATH = "peppers.tiff"
    
    image = Image.open(IMGPATH)
    image = np.asarray(image) / 255
    image = image[::2, ::2]

    n = image.shape[0]
    print(f"Image size: {image.shape}")

    # Generate PSF
    if not RGB:
        psfCentered = gaussianPSF(n, 4)
    else:
        psfCentered = np.stack([psfCentered for i in range(3)], axis=-1)
        psfCentered[...,0] = outOfFocusPSF(n,1)
        psfCentered[...,1] = outOfFocusPSF(n,6)
        psfCentered[...,2] = outOfFocusPSF(n,10)
        print(f"PSF size: {psfCentered.shape}")
    # Center PSF
    psf = np.roll(psfCentered, (-psfCentered.shape[0] // 2, -psfCentered.shape[0] // 2), axis=(0, 1))
    
    blurred, *_ = generateBlurredImage(image, noisePercent=.2, psf=psf, savePath=".npz/Blurred.npz")

    plot = True
    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        if not RGB:
            cmap = 'gray'
        else:
            cmap = None
        axs[0].imshow(image, cmap=cmap, vmin=0, vmax=1)
        axs[0].set_title("Original")
        axs[1].imshow(psfCentered/psfCentered.max(), cmap=cmap)
        axs[1].set_title("PSF")
        axs[2].imshow(blurred, cmap=cmap, vmin=0, vmax=1)
        axs[2].set_title("Blurred")
        plt.show()

