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

with np.load('grayscalePNPD.npz') as data:
    imRecPNPD = data['imRecPNPD']
    rreListPNPD = addIt0(data['rreListPNPD'], bRRE)
    ssimListPNPD = addIt0(data['ssimListPNPD'], bSSIM)
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

# Plot RRE vs Iteration: PNPD vs PNPD_NM
plotLists([rreListPNPD, rreListPNPD_NM],
            stopIndices=[dpStopIndexPNPD, dpStopIndexPNPD_NM],
            labels=["PNPD", "PNPD without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='RRE PNPD vs PNPD without momentum',
            xlabel='Iteration', ylabel='RRE',
            saveName='grayRRE_PNPDvsPNPD_NM_IT.pdf', semilogy=True)

# Plot RRE vs Time: PNPD vs PNPD_NM
plotLists([rreListPNPD, rreListPNPD_NM],
            X=[timeListPNPD, timeListPNPD_NM],
            stopIndices=[dpStopIndexPNPD, dpStopIndexPNPD_NM],
            labels=["PNPD", "PNPD without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='RRE PNPD vs PNPD without momentum',
            xlabel='Time (seconds)', ylabel='RRE',
            saveName='grayRRE_PNPDvsPNPD_NM_TIME.pdf', semilogy=True)

# Plot SSIM vs Iteration: PNPD vs PNPD_NM
plotLists([ssimListPNPD, ssimListPNPD_NM],
            stopIndices=[dpStopIndexPNPD, dpStopIndexPNPD_NM],
            labels=["PNPD", "PNPD without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='SSIM PNPD vs PNPD without momentum',
            xlabel='Iteration', ylabel='SSIM',
            saveName='graySSIM_PNPDvsPNPD_NM_IT.pdf')

# Plot SSIM vs Time: PNPD vs PNPD_NM
plotLists([ssimListPNPD, ssimListPNPD_NM],
            X=[timeListPNPD, timeListPNPD_NM],
            stopIndices=[dpStopIndexPNPD, dpStopIndexPNPD_NM],
            labels=["PNPD", "PNPD without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='SSIM PNPD vs PNPD without momentum',
            xlabel='Time (seconds)', ylabel='SSIM',
            saveName='graySSIM_PNPDvsPNPD_NM_TIME.pdf')

# Plot gamma vs Iteration: gammaFFBS vs gammaPNPD vs gammaPPNPD
# plotLists([gammaListPNPD, gammaFFBSListPNPD],
#             labels=["$\gamma^{PNPD}$", "$\gamma^{FFBS}$"],
#             title='$\gamma^{FFBS}$ vs $\gamma^{PNPD}$',
#             xlabel='Iteration', ylabel='$\gamma$',
#             linestyle=['-', '-.'],
#             saveName='grayGamma_IT.pdf')

with np.load('grayscalePNPD_K.npz') as data:
    x1_Kl = data['x1_K']
    imRecPNPD_Kl = data['imRecPNPD_K']
    rreListPNPD_Kl = data['rreListPNPD_K']
    ssimListPNPD_Kl = data['ssimListPNPD_K']
    timeListPNPD_Kl = data['timeListPNPD_K']
    dpStopIndexPNPD_Kl = data['dpStopIndexPNPD_K']
    kMaxValues = data['kMaxValues']
    lamValues  = data['lamValues']

    kMaxValues = [k for k in kMaxValues]

    rreListPNPD_Kl = [rreListPNPD_K for rreListPNPD_K in rreListPNPD_Kl]
    ssimListPNPD_Kl = [ssimListPNPD_K for ssimListPNPD_K in ssimListPNPD_Kl]
    timeListPNPD_Kl = [timeListPNPD_K for timeListPNPD_K in timeListPNPD_Kl]

    for i in range(len(lamValues)):
        rreListPNPD_Kl[i] = [addIt0(rreListPNPD_Kl[i][j], bRRE) for j in range(len(kMaxValues))]
        ssimListPNPD_Kl[i] = [addIt0(ssimListPNPD_Kl[i][j], bSSIM) for j in range(len(kMaxValues))]
        timeListPNPD_Kl[i] = [relTimetoAbsTime(timeListPNPD_Kl[i][j]) for j in range(len(kMaxValues))]

for i in range(len(lamValues)):
    # Plot RRE vs Iteration: PNPD Kmax
    plotLists(rreListPNPD_Kl[i],
                labels=[f"PNPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                title=f'RRE PNPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                xlabel='Iteration', ylabel='RRE',
                saveName=f'grayRRE_PNPD_K_l{lamValues[i]}_IT.pdf', semilogy=True)
    
    # Plot RRE vs Time: PNPD Kmax
    plotLists(rreListPNPD_Kl[i],
                X=timeListPNPD_Kl[i],
                labels=[f"PNPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                title=f'RRE PNPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                xlabel='Time (seconds)', ylabel='RRE',
                saveName=f'grayRRE_PNPD_K_l{lamValues[i]}_TIME.pdf', semilogy=True)
    
    # Plot SSIM vs Iteration: PNPD Kmax
    plotLists(ssimListPNPD_Kl[i],
                labels=[f"PNPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                title=f'SSIM PNPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                xlabel='Iteration', ylabel='SSIM',
                saveName=f'graySSIM_PNPD_K_l{lamValues[i]}_IT.pdf')
    
    # Plot SSIM vs Time: PNPD Kmax
    plotLists(ssimListPNPD_Kl[i],
                X=timeListPNPD_Kl[i],
                labels=[f"PNPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                title=f'SSIM PNPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                xlabel='Time (seconds)', ylabel='SSIM',
                saveName=f'graySSIM_PNPD_K_l{lamValues[i]}_TIME.pdf')

if show:
    plt.show()
