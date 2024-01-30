"""
NPDIT implementation

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

with np.load('grayscaleNPDIT.npz') as data:
    imRecNPDIT = data['imRecNPDIT']
    rreListNPDIT = addIt0(data['rreListNPDIT'], bRRE)
    ssimListNPDIT = addIt0(data['ssimListNPDIT'], bSSIM)
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

# Plot RRE vs Iteration: NPDIT vs NPDIT_NM
plotLists([rreListNPDIT, rreListNPDIT_NM],
            stopIndices=[dpStopIndexNPDIT, dpStopIndexNPDIT_NM],
            labels=["NPDIT", "NPDIT without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='RRE NPDIT vs NPDIT without momentum',
            xlabel='Iteration', ylabel='RRE',
            saveName='grayRRE_NPDITvsNPDIT_NM_IT.pdf', semilogy=True)

# Plot RRE vs Time: NPDIT vs NPDIT_NM
plotLists([rreListNPDIT, rreListNPDIT_NM],
            X=[timeListNPDIT, timeListNPDIT_NM],
            stopIndices=[dpStopIndexNPDIT, dpStopIndexNPDIT_NM],
            labels=["NPDIT", "NPDIT without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='RRE NPDIT vs NPDIT without momentum',
            xlabel='Time (seconds)', ylabel='RRE',
            saveName='grayRRE_NPDITvsNPDIT_NM_TIME.pdf', semilogy=True)

# Plot SSIM vs Iteration: NPDIT vs NPDIT_NM
plotLists([ssimListNPDIT, ssimListNPDIT_NM],
            stopIndices=[dpStopIndexNPDIT, dpStopIndexNPDIT_NM],
            labels=["NPDIT", "NPDIT without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='SSIM NPDIT vs NPDIT without momentum',
            xlabel='Iteration', ylabel='SSIM',
            saveName='graySSIM_NPDITvsNPDIT_NM_IT.pdf')

# Plot SSIM vs Time: NPDIT vs NPDIT_NM
plotLists([ssimListNPDIT, ssimListNPDIT_NM],
            X=[timeListNPDIT, timeListNPDIT_NM],
            stopIndices=[dpStopIndexNPDIT, dpStopIndexNPDIT_NM],
            labels=["NPDIT", "NPDIT without momentum"],
            labelsStop=['Discrepancy principle stop' for _ in range(2)],
            title='SSIM NPDIT vs NPDIT without momentum',
            xlabel='Time (seconds)', ylabel='SSIM',
            saveName='graySSIM_NPDITvsNPDIT_NM_TIME.pdf')

# Plot gamma vs Iteration: gammaFFBS vs gammaNPDIT vs gammaPNPDIT
# plotLists([gammaListNPDIT, gammaFFBSListNPDIT],
#             labels=["$\gamma^{NPDIT}$", "$\gamma^{FFBS}$"],
#             title='$\gamma^{FFBS}$ vs $\gamma^{NPDIT}$',
#             xlabel='Iteration', ylabel='$\gamma$',
#             linestyle=['-', '-.'],
#             saveName='grayGamma_IT.pdf')

# with np.load('grayscaleNPDIT_K.npz') as data:
#     x1_Kl = data['x1_K']
#     imRecNPDIT_Kl = data['imRecNPDIT_K']
#     rreListNPDIT_Kl = data['rreListNPDIT_K']
#     ssimListNPDIT_Kl = data['ssimListNPDIT_K']
#     timeListNPDIT_Kl = data['timeListNPDIT_K']
#     dpStopIndexNPDIT_Kl = data['dpStopIndexNPDIT_K']
#     kMaxValues = data['kMaxValues']
#     lamValues  = data['lamValues']

#     kMaxValues = [k for k in kMaxValues]

#     rreListNPDIT_Kl = [rreListNPDIT_K for rreListNPDIT_K in rreListNPDIT_Kl]
#     ssimListNPDIT_Kl = [ssimListNPDIT_K for ssimListNPDIT_K in ssimListNPDIT_Kl]
#     timeListNPDIT_Kl = [timeListNPDIT_K for timeListNPDIT_K in timeListNPDIT_Kl]

#     for i in range(len(lamValues)):
#         rreListNPDIT_Kl[i] = [addIt0(rreListNPDIT_Kl[i][j], bRRE) for j in range(len(kMaxValues))]
#         ssimListNPDIT_Kl[i] = [addIt0(ssimListNPDIT_Kl[i][j], bSSIM) for j in range(len(kMaxValues))]
#         timeListNPDIT_Kl[i] = [relTimetoAbsTime(timeListNPDIT_Kl[i][j]) for j in range(len(kMaxValues))]

# for i in range(len(lamValues)):
#     # Plot RRE vs Iteration: NPDIT Kmax
#     plotLists(rreListNPDIT_Kl[i],
#                 labels=[f"NPDIT $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
#                 title=f'RRE NPDIT $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
#                 xlabel='Iteration', ylabel='RRE',
#                 saveName=f'grayRRE_NPDIT_K_l{lamValues[i]}_IT.pdf', semilogy=True)
    
#     # Plot RRE vs Time: NPDIT Kmax
#     plotLists(rreListNPDIT_Kl[i],
#                 X=timeListNPDIT_Kl[i],
#                 labels=[f"NPDIT $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
#                 title=f'RRE NPDIT $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
#                 xlabel='Time (seconds)', ylabel='RRE',
#                 saveName=f'grayRRE_NPDIT_K_l{lamValues[i]}_TIME.pdf', semilogy=True)
    
#     # Plot SSIM vs Iteration: NPDIT Kmax
#     plotLists(ssimListNPDIT_Kl[i],
#                 labels=[f"NPDIT $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
#                 title=f'SSIM NPDIT $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
#                 xlabel='Iteration', ylabel='SSIM',
#                 saveName=f'graySSIM_NPDIT_K_l{lamValues[i]}_IT.pdf')
    
#     # Plot SSIM vs Time: NPDIT Kmax
#     plotLists(ssimListNPDIT_Kl[i],
#                 X=timeListNPDIT_Kl[i],
#                 labels=[f"NPDIT $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
#                 title=f'SSIM NPDIT $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
#                 xlabel='Time (seconds)', ylabel='SSIM',
#                 saveName=f'graySSIM_NPDIT_K_l{lamValues[i]}_TIME.pdf')

if show:
    plt.show()
