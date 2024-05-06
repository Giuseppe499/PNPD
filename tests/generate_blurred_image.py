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

from PIL import Image
import numpy as np
from numpy.fft import fft2
from mathExtras import convolve_2D_fft, generate_gaussian_PSF, generate_out_of_focus_PSF, scalar_product
import os


def generateBlurredImage(image, noisePercent, psf, savePath=None):
    # Generate blurred image
    conv = convolve_2D_fft(image, psf)

    # Generate noise
    noise = np.random.normal(size=image.shape)
    noise *= noisePercent * np.linalg.norm(conv) / np.linalg.norm(noise)
    noiseNormSqd = scalar_product(noise.ravel())

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
        np.savez(
            savePath,
            conv=conv,
            noise=noise,
            b=b,
            bFFT=bFFT,
            psf=psf,
            psfFFT=psfFFT,
            psfFFTC=psfFFTC,
            psfAbsSq=psfAbsSq,
            image=image,
            noiseNormSqd=noiseNormSqd,
        )

    return (
        conv,
        noise,
        b,
        bFFT,
        psf,
        psfFFT,
        psfFFTC,
        psfAbsSq,
        image,
        noiseNormSqd,
    )


if __name__ == "__main__":
    RGB = False

    if not RGB:
        IMGPATH = "cameraman.tif"
    else:
        IMGPATH = "peppers.tiff"

    np.random.seed(42)

    image = Image.open(IMGPATH)
    image = np.asarray(image) / 255
    image = image[::2, ::2]

    n = image.shape[0]
    print(f"Image size: {image.shape}")

    # Generate PSF
    if not RGB:
        psfCentered = generate_gaussian_PSF(n, 2)
    else:
        psfCentered = np.stack([psfCentered for i in range(3)], axis=-1)
        psfCentered[..., 0] = generate_out_of_focus_PSF(n, 1)
        psfCentered[..., 1] = generate_out_of_focus_PSF(n, 6)
        psfCentered[..., 2] = generate_out_of_focus_PSF(n, 10)
        print(f"PSF size: {psfCentered.shape}")

    # Move PSF center to the top-left corner
    psf = np.roll(
        psfCentered,
        (-psfCentered.shape[0] // 2, -psfCentered.shape[0] // 2),
        axis=(0, 1),
    )

    blurred = generateBlurredImage(
        image, noisePercent=0.01, psf=psf, savePath="./npz/Blurred.npz"
    )[2]

    plot = True
    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        if not RGB:
            cmap = "gray"
        else:
            cmap = None
        axs[0].imshow(image, cmap=cmap, vmin=0, vmax=1)
        axs[0].set_title("Original")
        axs[1].imshow(psfCentered / psfCentered.max(), cmap=cmap)
        axs[1].set_title("PSF")
        axs[2].imshow(blurred, cmap=cmap, vmin=0, vmax=1)
        axs[2].set_title("Blurred")
        plt.show()
