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
import matplotlib
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import grayConfig

def main(suffixNPD, suffixPNPD, suffixNPDIT, recIndexes):
    with np.load(f'./npz/{grayConfig.prefix}/Blurred.npz') as data:
        b = data['b']
        psfBT = data['psfBT']
        image = data['image']
        conv = data['conv']
        noise = data['noise']

    with np.load(f'./npz/{grayConfig.prefix}/NPD_{suffixNPD}.npz') as data:
        imRecNPD = data['imRec']
        dpStopIndexNPD = data['dpStopIndex']
        recListNPD = data['recList']

        gammaListNPD = data['gammaList']
        gammaFFBSListNPD = data['gammaFFBSList']

    with np.load(f'./npz/{grayConfig.prefix}/PNPD_{suffixPNPD}.npz') as data:
        imRecPNPD = data['imRec']
        dpStopIndexPNPD = data['dpStopIndex']
        recListPNPD = data['recList']

    with np.load(f'./npz/{grayConfig.prefix}/NPDIT_{suffixNPDIT}.npz') as data:
        imRecNPDIT = data['imRec']
        dpStopIndexNPDIT = data['dpStopIndex']
        recListNPDIT = data['recList']
    

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

    matplotlib.rcParams["font.size"] = 10

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
    ax[2].set_title('$\\tilde b = x \circledast_{2D}$ PSF')
    plt.savefig(savePath(f'{grayConfig.prefix}/PSFConv.pdf'), bbox_inches='tight', dpi=1200)

    # Plot conv + noise = blurred image
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5)
    for axis in ax:
        axis.axis('off')
    ax[0].imshow(conv, cmap='gray', vmin = 0, vmax = 1)
    ax[0].set_title('$\\tilde b$')
    ax[0].text(1.15, .5, "$+$", fontsize=20, transform = ax[0].transAxes)
    ax[1].imshow(noise, cmap='gray', vmin = noise.min()/2, vmax = noise.max()/2)
    ax[1].set_title('$e$')
    ax[1].text(1.15, .5, "$=$", fontsize=20, transform = ax[1].transAxes)
    ax[2].imshow(b, cmap='gray', vmin = 0, vmax = 1)
    ax[2].set_title('$b = \\tilde b + e$')
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
    ax[1].imshow(psfEX, cmap='gray')
    ax[1].set_title(f'$u$')
    ax[1].text(1.15, .5, "$=$", fontsize=20, transform = ax[1].transAxes)
    ax[2].imshow(convEX, cmap='gray', vmin = 0, vmax = 1)
    ax[2].set_title('$x \circledast_{2D}$ u')
    plt.savefig(savePath(f'{grayConfig.prefix}/PSFConvExample.pdf'), bbox_inches='tight', dpi=1200)

    matplotlib.rcParams["font.size"] = 12

    def plotReconstruction(imRec, title, iteration = None):
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
        ax.text(boxPos[0], boxPos[1]+vOffset, text, bbox=props, transform = ax.transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')

    def plotRecList(recList, recIndexes, recDP, itDP, method):
        n = len(recList) + 1
        plt.figure(figsize=(5*n, 5))
        plt.subplot(1, n, 1)
        plotReconstruction(recDP, f'{method} discrepancy principle', iteration = itDP)
        for i, rec in enumerate(recList):
            plt.subplot(1, n, i+2)
            plotReconstruction(rec, f'{method} iteration {recIndexes[i]}', iteration = recIndexes[i])

    
    # Plot blurred image
    plt.figure()
    plotReconstruction(b, 'Blurred Image')
    plt.savefig(savePath(f'{grayConfig.prefix}/BlurredImage.pdf'), bbox_inches='tight', dpi=300)

    # Plot NPD reconstruction
    plotRecList(recListNPD, recIndexes, imRecNPD, dpStopIndexNPD, 'NPD')
    n = len(recIndexes) + 1
    plt.savefig(savePath(f'{grayConfig.prefix}/NPD_Reconstruction.pdf'), bbox_inches='tight', dpi=300*n)

    # Plot PNPD reconstruction
    plotRecList(recListPNPD, recIndexes, imRecPNPD, dpStopIndexPNPD, 'PNPD')
    plt.savefig(savePath(f'{grayConfig.prefix}/PNPD_Reconstruction.pdf'), bbox_inches='tight', dpi=300*n)

    # Plot NPDIT reconstruction
    plotRecList(recListNPDIT, recIndexes, imRecNPDIT, dpStopIndexNPDIT, 'NPDIT')
    plt.savefig(savePath(f'{grayConfig.prefix}/NPDIT_Reconstruction.pdf'), bbox_inches='tight', dpi=300*n)

    # Plot gamma vs Iteration: gammaFFBS vs gammaNPD
    plotLists([gammaListNPD, gammaFFBSListNPD],
                labels=["$\gamma^{NPD}$", "$\gamma^{FFBS}$"],
                title='$\gamma^{FFBS}$ vs $\gamma^{NPD}$',
                xlabel='Iteration', ylabel='$\gamma$',
                linestyle=['-', '-.'],
                saveName=f'{grayConfig.prefix}/Gamma_IT.pdf')

    if show:
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main() 