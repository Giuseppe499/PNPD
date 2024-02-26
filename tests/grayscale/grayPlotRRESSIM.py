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

def main(filenameList, nameList, saveNameList=None, saveStr=None, title=None, showStop = True, show=False):
    if saveNameList is None:
        saveNameList = nameList
    with np.load(f'./npz/{grayConfig.prefix}/Blurred.npz') as data:
        b = data['b']
        image = data['image']
        bSSIM = ssim(b, image, data_range=1)
        bRRE = np.linalg.norm(b - image) / np.linalg.norm(image)

    rre = []
    SSIM = []
    time = []
    dpStopIndex = []
    for filename in filenameList:
        with np.load(f'./npz/{grayConfig.prefix}/{filename}.npz') as data:
            rre.append(addIt0(data['rreList'], bRRE))
            SSIM.append(addIt0(data['ssimList'], bSSIM))
            time.append(relTimetoAbsTime(data['timeList']))
            dpStopIndex.append(data['dpStopIndex'])
    
    if title is None:
        title = " vs ".join(nameList)
    if saveStr is None:
        saveStr = "vs".join(saveNameList)
    
    labelsStop = ['Discrepancy principle stop' for _ in range(len(nameList))]
    if not showStop:
        dpStopIndex = None
    # Plot RRE vs Iteration
    plotLists(rre, stopIndices=dpStopIndex, labels=nameList,
                labelsStop=labelsStop,
                title=f'RRE {title}',
                xlabel='Iteration', ylabel='RRE',
                saveName=f'{grayConfig.prefix}/RRE_{saveStr}_IT.pdf', semilogy=True)

    # Plot RRE vs Time
    plotLists(rre, X=time, stopIndices=dpStopIndex, labels=nameList,
                labelsStop=labelsStop,
                title=f'RRE {title}',
                xlabel='Time (seconds)', ylabel='RRE',
                saveName=f'{grayConfig.prefix}/RRE_{saveStr}_TIME.pdf', semilogy=True)
    
    # Plot SSIM vs Iteration
    plotLists(SSIM, stopIndices=dpStopIndex, labels=nameList,
                labelsStop=labelsStop,
                title=f'SSIM {title}',
                xlabel='Iteration', ylabel='SSIM',
                saveName=f'{grayConfig.prefix}/SSIM_{saveStr}_IT.pdf')
    
    # Plot SSIM vs Time
    plotLists(SSIM, X=time, stopIndices=dpStopIndex, labels=nameList,
                labelsStop=labelsStop,
                title=f'SSIM {title}',
                xlabel='Time (seconds)', ylabel='SSIM',
                saveName=f'{grayConfig.prefix}/SSIM_{saveStr}_TIME.pdf')

    if show:
        plt.show()
    else:
        plt.close('all')