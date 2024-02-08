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
import grayConfig

def main():
    with np.load(f'./npz/{grayConfig.prefix}/Blurred.npz') as data:
        b = data['b']
        psf = data['psf']
        image = data['image']
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

    show = False

    # Plot RRE vs Iteration: NPD vs NPD_NM
    plotLists([rreListNPD, rreListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='RRE NPD vs NPD without momentum',
                xlabel='Iteration', ylabel='RRE',
                saveName=f'{grayConfig.prefix}/RRE_NPDvsNPD_NM_IT.pdf', semilogy=True)

    # Plot RRE vs Time: NPD vs NPD_NM
    plotLists([rreListNPD, rreListNPD_NM],
                X=[timeListNPD, timeListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='RRE NPD vs NPD without momentum',
                xlabel='Time (seconds)', ylabel='RRE',
                saveName=f'{grayConfig.prefix}/RRE_NPDvsNPD_NM_TIME.pdf', semilogy=True)

    # Plot SSIM vs Iteration: NPD vs NPD_NM
    plotLists([ssimListNPD, ssimListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='SSIM NPD vs NPD without momentum',
                xlabel='Iteration', ylabel='SSIM',
                saveName=f'{grayConfig.prefix}/SSIM_NPDvsNPD_NM_IT.pdf')

    # Plot SSIM vs Time: NPD vs NPD_NM
    plotLists([ssimListNPD, ssimListNPD_NM],
                X=[timeListNPD, timeListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='SSIM NPD vs NPD without momentum',
                xlabel='Time (seconds)', ylabel='SSIM',
                saveName=f'{grayConfig.prefix}/SSIM_NPDvsNPD_NM_TIME.pdf')

    # Plot gamma vs Iteration: gammaFFBS vs gammaNPD vs gammaPNPD
    plotLists([gammaListNPD, gammaFFBSListNPD],
                labels=["$\gamma^{NPD}$", "$\gamma^{FFBS}$"],
                title='$\gamma^{FFBS}$ vs $\gamma^{NPD}$',
                xlabel='Iteration', ylabel='$\gamma$',
                linestyle=['-', '-.'],
                saveName=f'{grayConfig.prefix}/Gamma_IT.pdf')

    with np.load(f'./npz/{grayConfig.prefix}/NPD_K.npz') as data:
        x1_Kl = data['x1_K']
        imRecNPD_Kl = data['imRecNPD_K']
        rreListNPD_Kl = data['rreListNPD_K']
        ssimListNPD_Kl = data['ssimListNPD_K']
        timeListNPD_Kl = data['timeListNPD_K']
        dpStopIndexNPD_Kl = data['dpStopIndexNPD_K']
        kMaxValues = data['kMaxValues']
        lamValues  = data['lamValues']

        kMaxValues = [k for k in kMaxValues]

        rreListNPD_Kl = [rreListNPD_K for rreListNPD_K in rreListNPD_Kl]
        ssimListNPD_Kl = [ssimListNPD_K for ssimListNPD_K in ssimListNPD_Kl]
        timeListNPD_Kl = [timeListNPD_K for timeListNPD_K in timeListNPD_Kl]

        for i in range(len(lamValues)):
            rreListNPD_Kl[i] = [addIt0(rreListNPD_Kl[i][j], bRRE) for j in range(len(kMaxValues))]
            ssimListNPD_Kl[i] = [addIt0(ssimListNPD_Kl[i][j], bSSIM) for j in range(len(kMaxValues))]
            timeListNPD_Kl[i] = [relTimetoAbsTime(timeListNPD_Kl[i][j]) for j in range(len(kMaxValues))]

    for i in range(len(lamValues)):
        # Plot RRE vs Iteration: NPD Kmax
        plotLists(rreListNPD_Kl[i],
                    labels=[f"NPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                    title=f'RRE NPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                    xlabel='Iteration', ylabel='RRE',
                    saveName=f'{grayConfig.prefix}/RRE_NPD_K_l{lamValues[i]}_IT.pdf', semilogy=True)
        
        # Plot RRE vs Time: NPD Kmax
        plotLists(rreListNPD_Kl[i],
                    X=timeListNPD_Kl[i],
                    labels=[f"NPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                    title=f'RRE NPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                    xlabel='Time (seconds)', ylabel='RRE',
                    saveName=f'{grayConfig.prefix}/RRE_NPD_K_l{lamValues[i]}_TIME.pdf', semilogy=True)
        
        # Plot SSIM vs Iteration: NPD Kmax
        plotLists(ssimListNPD_Kl[i],
                    labels=[f"NPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                    title=f'SSIM NPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                    xlabel='Iteration', ylabel='SSIM',
                    saveName=f'{grayConfig.prefix}/SSIM_NPD_K_l{lamValues[i]}_IT.pdf')
        
        # Plot SSIM vs Time: NPD Kmax
        plotLists(ssimListNPD_Kl[i],
                    X=timeListNPD_Kl[i],
                    labels=[f"NPD $k_{{max}}$ = {kMax}" for kMax in kMaxValues],
                    title=f'SSIM NPD $k_{{max}}$ = {kMaxValues}, $\lambda$ = {lamValues[i]}',
                    xlabel='Time (seconds)', ylabel='SSIM',
                    saveName=f'{grayConfig.prefix}/SSIM_NPD_K_l{lamValues[i]}_TIME.pdf')

    if show:
        plt.show()

if __name__ == "__main__":
    main()
