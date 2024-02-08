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

from tests.plotExtras import *
from mathExtras import (centerCrop, gaussianPSF, fftConvolve2D)
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import grayConfig

def main():
    with np.load(f'./npz/{grayConfig.prefix}/Blurred.npz') as data:
        b = data['b']
        psf = data['psf']
        psfBT = data['psfBT']
        image = data['image']
        conv = data['conv']
        noise = data['noise']
        bSSIM = ssim(b, image, data_range=1)
        bRRE = np.linalg.norm(b - image) / np.linalg.norm(image)

    with np.load(f'./npz/{grayConfig.prefix}/NPD.npz') as data:
        imRecNPD = data['imRecNPD']
        rreListNPD = addIt0(data['rreListNPD'], bRRE)
        ssimListNPD = addIt0(data['ssimListNPD'], bSSIM)
        timeListNPD = relTimetoAbsTime(data['timeListNPD'])
        dpStopIndexNPD = data['dpStopIndexNPD']
        rreRecNPD = rreListNPD[dpStopIndexNPD]
        ssimRecNPD = ssimListNPD[dpStopIndexNPD]

        imRecNPD_NM = data['imRecNPD_NM']
        rreListNPD_NM = addIt0(data['rreListNPD_NM'], bRRE)
        ssimListNPD_NM = addIt0(data['ssimListNPD_NM'], bSSIM)
        timeListNPD_NM = relTimetoAbsTime(data['timeListNPD_NM'])
        dpStopIndexNPD_NM = data['dpStopIndexNPD_NM']

        gammaListNPD = data['gammaListNPD']
        gammaFFBSListNPD = data['gammaFFBSListNPD']

    with np.load(f'./npz/{grayConfig.prefix}/PNPD.npz') as data:
        imRecPNPD = data['imRecPNPD']
        rreListPNPD = addIt0(data['rreListPNPD'], bRRE)
        ssimListPNPD = addIt0(data['ssimListPNPD'], bSSIM)
        ssimPNPD = ssim(imRecPNPD, image, data_range=1)
        timeListPNPD = relTimetoAbsTime(data['timeListPNPD'])
        dpStopIndexPNPD = data['dpStopIndexPNPD']
        rreRecPNPD = rreListPNPD[dpStopIndexPNPD]
        ssimRecPNPD = ssimListPNPD[dpStopIndexPNPD]

        imRecPNPD_NM = data['imRecPNPD_NM']
        rreListPNPD_NM = addIt0(data['rreListPNPD_NM'], bRRE)
        ssimListPNPD_NM = addIt0(data['ssimListPNPD_NM'], bSSIM)
        timeListPNPD_NM = relTimetoAbsTime(data['timeListPNPD_NM'])
        dpStopIndexPNPD_NM = data['dpStopIndexPNPD_NM']

        gammaListPNPD = data['gammaListPNPD']
        gammaFFBSListPNPD = data['gammaFFBSListPNPD']

    with np.load(f'./npz/{grayConfig.prefix}/NPDIT.npz') as data:
        imRecNPDIT = data['imRecNPDIT']
        rreListNPDIT = addIt0(data['rreListNPDIT'], bRRE)
        ssimListNPDIT = addIt0(data['ssimListNPDIT'], bSSIM)
        ssimNPDIT = ssim(imRecNPDIT, image, data_range=1)
        timeListNPDIT = relTimetoAbsTime(data['timeListNPDIT'])
        dpStopIndexNPDIT = data['dpStopIndexNPDIT']
        rreRecNPDIT = rreListNPDIT[dpStopIndexNPDIT]
        ssimRecNPDIT = ssimListNPDIT[dpStopIndexNPDIT]

        imRecNPDIT_NM = data['imRecNPDIT_NM']
        rreListNPDIT_NM = addIt0(data['rreListNPDIT_NM'], bRRE)
        ssimListNPDIT_NM = addIt0(data['ssimListNPDIT_NM'], bSSIM)
        timeListNPDIT_NM = relTimetoAbsTime(data['timeListNPDIT_NM'])
        dpStopIndexNPDIT_NM = data['dpStopIndexNPDIT_NM']

        gammaListNPDIT = data['gammaListNPDIT']
        gammaFFBSListNPDIT = data['gammaFFBSListNPDIT']
    

    show = False

    # Plot original image
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Original Image')
    filename = savePath(f'{grayConfig.prefix}/OriginalImage.pdf')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', dpi=600)

    # Plot PSF
    cropSize = 20
    plt.figure()
    plt.axis('off')
    plt.imshow(centerCrop(psfBT, (cropSize, cropSize)), cmap='gray')
    plt.title(f'PSF (cropped to {cropSize}x{cropSize})')
    plt.savefig(savePath(f'{grayConfig.prefix}/PSF.pdf'), bbox_inches='tight', dpi=300)

    # Plot image * PSF = conv
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5)
    for axis in ax:
        axis.axis('off')
    ax[0].imshow(image, cmap='gray', vmin = 0, vmax = 1)
    ax[0].set_title('$x$')
    ax[0].text(1.05, .5, "$\circledast_{2D}$", fontsize=20, transform = ax[0].transAxes)
    ax[1].imshow(centerCrop(psfBT, (cropSize, cropSize)), cmap='gray')
    ax[1].set_title(f'PSF\n(center crop of size {cropSize}x{cropSize})')
    ax[1].text(1.15, .5, "$=$", fontsize=20, transform = ax[1].transAxes)
    ax[2].imshow(conv, cmap='gray', vmin = 0, vmax = 1)
    ax[2].set_title('$b = x \circledast_{2D}$ PSF')
    plt.savefig(savePath(f'{grayConfig.prefix}/PSFConv.pdf'), bbox_inches='tight', dpi=1200)

    # Plot conv + noise = blurred image
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5)
    for axis in ax:
        axis.axis('off')
    ax[0].imshow(conv, cmap='gray', vmin = 0, vmax = 1)
    ax[0].set_title('$b$')
    ax[0].text(1.15, .5, "$+$", fontsize=20, transform = ax[0].transAxes)
    ax[1].imshow(noise, cmap='gray', vmin = noise.min()/2, vmax = noise.max()/2)
    ax[1].set_title('$n$')
    ax[1].text(1.15, .5, "$=$", fontsize=20, transform = ax[1].transAxes)
    ax[2].imshow(b, cmap='gray', vmin = 0, vmax = 1)
    ax[2].set_title('$\\tilde b = b + n$')
    plt.savefig(savePath(f'{grayConfig.prefix}/BlurredPlusNoise.pdf'), bbox_inches='tight', dpi=1200)

    # Plot image * PSF = conv Convolution example
    psfEXBT = gaussianPSF(image.shape[0], 10)
    psfEX = np.roll(psfEXBT, (-psfEXBT.shape[0] // 2, -psfEXBT.shape[0] // 2), axis=(0, 1))
    convEX = fftConvolve2D(image, psfEX)
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5)
    for axis in ax:
        axis.axis('off')
    ax[0].imshow(image, cmap='gray', vmin = 0, vmax = 1)
    ax[0].set_title('$x$')
    ax[0].text(1.05, .5, "$\circledast_{2D}$", fontsize=20, transform = ax[0].transAxes)
    ax[1].imshow(psfEXBT, cmap='gray')
    ax[1].set_title(f'$u$')
    ax[1].text(1.15, .5, "$=$", fontsize=20, transform = ax[1].transAxes)
    ax[2].imshow(convEX, cmap='gray', vmin = 0, vmax = 1)
    ax[2].set_title('$x \circledast_{2D}$ u')
    plt.savefig(savePath(f'{grayConfig.prefix}/PSFConvExample.pdf'), bbox_inches='tight', dpi=1200)

    def plotReconstruction(imRec, title, iteration = None):
        plt.figure()
        plt.axis('off')
        plt.imshow(imRec, cmap='gray', vmin = 0, vmax = 1)
        plt.title(title)
        ax = plt.gca()
        SSIM = ssim(imRec, image, data_range=1)
        rre = np.linalg.norm(imRec - image) / np.linalg.norm(image)
        vOffset = 0
        if iteration is not None:
            text = f"Iteration: {iteration}\nRRE: {rre:.4f}\nSSIM: {SSIM:.4f}"
            vOffset = -0.05
        else:
            text = f"RRE: {rre:.4f}\nSSIM: {SSIM:.4f}"
        ax.text(boxPos[0], boxPos[1]+vOffset, text, bbox=props, transform = ax.transAxes, fontsize=12)
        
    # Plot blurred image
    plotReconstruction(b, 'Blurred Image')
    plt.savefig(savePath(f'{grayConfig.prefix}/BlurredImage.pdf'), bbox_inches='tight', dpi=600)

    # Plot NPD reconstruction
    plotReconstruction(imRecNPD, 'NPD Reconstruction', iteration =  dpStopIndexNPD)
    plt.savefig(savePath(f'{grayConfig.prefix}/NPD_Reconstruction.pdf'), bbox_inches='tight', dpi=600)

    # Plot NPD without momentum reconstruction
    plotReconstruction(imRecNPD_NM, 'NPD without momentum Reconstruction', iteration =  dpStopIndexNPD_NM)
    plt.savefig(savePath(f'{grayConfig.prefix}/NPD_NM_Reconstruction.pdf'), bbox_inches='tight', dpi=600)

    # Plot PNPD reconstruction
    plotReconstruction(imRecPNPD, 'PNPD Reconstruction', iteration =  dpStopIndexPNPD)
    plt.savefig(savePath(f'{grayConfig.prefix}/PNPD_Reconstruction.pdf'), bbox_inches='tight', dpi=600)

    # Plot PNPD without momentum reconstruction
    plotReconstruction(imRecPNPD_NM, 'PNPD without momentum Reconstruction', iteration =  dpStopIndexPNPD_NM)
    plt.savefig(savePath(f'{grayConfig.prefix}/PNPD_NM_Reconstruction.pdf'), bbox_inches='tight', dpi=600)

    # Plot NPDIT reconstruction
    plotReconstruction(imRecNPDIT, 'NPDIT Reconstruction', iteration =  dpStopIndexNPDIT)
    plt.savefig(savePath(f'{grayConfig.prefix}/NPDIT_Reconstruction.pdf'), bbox_inches='tight', dpi=600)

    if show:
        plt.show()

if __name__ == "__main__":
    main() 