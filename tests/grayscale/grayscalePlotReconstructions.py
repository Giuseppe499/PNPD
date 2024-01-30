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
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

with np.load('grayscaleBlurred.npz') as data:
    b = data['b']
    psf = data['psf']
    image = data['image']
    conv = data['conv']
    noise = data['noise']
    bSSIM = ssim(b, image, data_range=1)
    bRRE = np.linalg.norm(b - image) / np.linalg.norm(image)

with np.load('grayscaleNPD.npz') as data:
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

with np.load('grayscalePNPD.npz') as data:
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

    

show = False

# Plot original image
plt.figure()
plt.axis('off')
plt.imshow(image, cmap='gray', vmin = 0, vmax = 1)
plt.title('Original Image')
plt.savefig(savePath('grayOriginalImage.pdf'), bbox_inches='tight')

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
plt.savefig(savePath('grayBlurredPlusNoise.pdf'), bbox_inches='tight')

def plotReconstruction(imRec, title):
    plt.figure()
    plt.axis('off')
    plt.imshow(imRec, cmap='gray', vmin = 0, vmax = 1)
    plt.title(title)
    ax = plt.gca()
    SSIM = ssim(imRec, image, data_range=1)
    rre = np.linalg.norm(imRec - image) / np.linalg.norm(image)
    ax.text(boxPos[0], boxPos[1], f"SSIM: {SSIM:.4f}\nRRE:  {rre:.4f}", bbox=props, transform = ax.transAxes, fontsize=12)
    
# Plot blurred image
plotReconstruction(b, 'Blurred Image')
plt.savefig(savePath('grayBlurredImage.pdf'), bbox_inches='tight')

# Plot NPD reconstruction
plotReconstruction(imRecNPD, 'NPD Reconstruction')
plt.savefig(savePath('grayNPD_Reconstruction.pdf'), bbox_inches='tight')

# Plot NPD without momentum reconstruction
plotReconstruction(imRecNPD_NM, 'NPD without momentum Reconstruction')
plt.savefig(savePath('grayNPD_NM_Reconstruction.pdf'), bbox_inches='tight')

# Plot PNPD reconstruction
plotReconstruction(imRecPNPD, 'PNPD Reconstruction')
plt.savefig(savePath('grayPNPD_Reconstruction.pdf'), bbox_inches='tight')

# Plot PNPD without momentum reconstruction
plotReconstruction(imRecPNPD_NM, 'PNPD without momentum Reconstruction')
plt.savefig(savePath('grayPNPD_NM_Reconstruction.pdf'), bbox_inches='tight')

# # Plot RRE vs Iteration: NPD vs PNPD
# plotLists([rreListNPD, rreListPNPD],
#             stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
#             labels=["NPD", "PNPD"],
#             labelsStop=['Discrepancy principle stop' for _ in range(2)],
#             title='RRE NPD vs PNPD',
#             xlabel='Iteration', ylabel='RRE',
#             saveName='grayRRE_NPDvsPNPD_IT.pdf', semilogy=True)

# # Plot RRE vs Time: NPD vs PNPD
# plotLists([rreListNPD, rreListPNPD],
#             X=[timeListNPD, timeListPNPD],
#             stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
#             labels=["NPD", "PNPD"],
#             labelsStop=['Discrepancy principle stop' for _ in range(2)],
#             title='RRE NPD vs PNPD',
#             xlabel='Time (seconds)', ylabel='RRE',
#             saveName='grayRRE_NPDvsPNPD_TIME.pdf', semilogy=True)

# # Plot SSIM vs Iteration: NPD vs PNPD
# plotLists([ssimListNPD, ssimListPNPD],
#             stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
#             labels=["NPD", "PNPD"],
#             labelsStop=['Discrepancy principle stop' for _ in range(2)],
#             title='SSIM NPD vs PNPD',
#             xlabel='Iteration', ylabel='SSIM',
#             saveName='graySSIM_NPDvsPNPD_IT.pdf')

# # Plot SSIM vs Time: NPD vs PNPD
# plotLists([ssimListNPD, ssimListPNPD],
#             X=[timeListNPD, timeListPNPD],
#             stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
#             labels=["NPD", "PNPD"],
#             labelsStop=['Discrepancy principle stop' for _ in range(2)],
#             title='SSIM NPD vs PNPD',
#             xlabel='Time (seconds)', ylabel='SSIM',
#             saveName='graySSIM_NPDvsPNPD_TIME.pdf')

if show:
    plt.show()
