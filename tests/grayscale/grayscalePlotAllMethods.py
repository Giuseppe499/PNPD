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

# Plot RRE vs Iteration: NPD vs PNPD
plotLists([rreListNPD, rreListPNPD],
            stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
            labels=["NPD", "PNPD"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='RRE NPD vs PNPD',
            xlabel='Iteration', ylabel='RRE',
            saveName='grayRRE_NPDvsPNPD_IT.pdf', semilogy=True)

# Plot RRE vs Time: NPD vs PNPD
plotLists([rreListNPD, rreListPNPD],
            X=[timeListNPD, timeListPNPD],
            stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
            labels=["NPD", "PNPD"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='RRE NPD vs PNPD',
            xlabel='Time (seconds)', ylabel='RRE',
            saveName='grayRRE_NPDvsPNPD_TIME.pdf', semilogy=True)

# Plot SSIM vs Iteration: NPD vs PNPD
plotLists([ssimListNPD, ssimListPNPD],
            stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
            labels=["NPD", "PNPD"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='SSIM NPD vs PNPD',
            xlabel='Iteration', ylabel='SSIM',
            saveName='graySSIM_NPDvsPNPD_IT.pdf')

# Plot SSIM vs Time: NPD vs PNPD
plotLists([ssimListNPD, ssimListPNPD],
            X=[timeListNPD, timeListPNPD],
            stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
            labels=["NPD", "PNPD"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='SSIM NPD vs PNPD',
            xlabel='Time (seconds)', ylabel='SSIM',
            saveName='graySSIM_NPDvsPNPD_TIME.pdf')

if show:
    plt.show()
