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

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

def savePath(filename):
        directory = "./Plots/"
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

def relTimetoAbsTime(timeList):
    absTimeList = np.zeros(len(timeList)+1)
    absTimeList[0] = 0
    absTimeList[1] = timeList[0]
    for i in range(1, len(timeList)):
        absTimeList[i+1] = absTimeList[i] + timeList[i]
    return absTimeList

def addIt0(list, value):
    newList = np.zeros(len(list)+1)
    newList[0] = value
    newList[1:] = list
    return newList

if __name__ == '__main__':
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

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    props = dict(boxstyle='square', fc="white", ec="black" , alpha=0.5)
    boxPos = (0.72, 0.895)

    # Plot original image
    plt.figure()
    plt.imshow(image, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Original Image')
    plt.savefig(savePath('grayOriginalImage.pdf'), bbox_inches='tight')

    def plotReconstruction(imRec, title):
        plt.figure()
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

    def plotLists(Y, X = None, stopIndices = None, labels = None, labelsStop = None, title = None, xlabel = None, ylabel = None, saveName = None, linestyle = None, semilogy = False):
        plt.figure()
        if X is None:
            X = [np.arange(len(Y[i])) for i in range(len(Y))]
        if labels is None:
            labels = ['' for i in range(len(Y))]
        if linestyle is None:
            linestyle = ['-' for i in range(len(Y))]
        for i in range(len(Y)):
            if semilogy:
                plt.semilogy(X[i], Y[i], label=labels[i], linestyle=linestyle[i])
            else:
                plt.plot(X[i], Y[i], label=labels[i], linestyle=linestyle[i])
        plt.gca().set_prop_cycle(None)
        if stopIndices is not None:
            for i in range(len(stopIndices)):
                plt.plot(X[i][stopIndices[i]], Y[i][stopIndices[i]], 'o', label=labelsStop[i])
        plt.legend()
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if saveName is not None:
            plt.savefig(savePath(saveName), bbox_inches='tight')

    # Plot RRE vs Iteration: NPD vs NPD_NM
    plotLists([rreListNPD, rreListNPD_NM],
              stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
              labels=["NPD", "NPD without momentum"],
              labelsStop=['Discrepancy principle stop' for _ in range(2)],
              title='RRE NPD vs NPD without momentum',
              xlabel='Iteration', ylabel='RRE',
              saveName='grayRRE_NPDvsNPD_NM_IT.pdf', semilogy=True)

    # Plot RRE vs Time: NPD vs NPD_NM
    plotLists([rreListNPD, rreListNPD_NM],
              X=[timeListNPD, timeListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='RRE NPD vs NPD without momentum',
                xlabel='Time (seconds)', ylabel='RRE',
                saveName='grayRRE_NPDvsNPD_NM_TIME.pdf', semilogy=True)

    # Plot SSIM vs Iteration: NPD vs NPD_NM
    plotLists([ssimListNPD, ssimListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='SSIM NPD vs NPD without momentum',
                xlabel='Iteration', ylabel='SSIM',
                saveName='graySSIM_NPDvsNPD_NM_IT.pdf')

    # Plot SSIM vs Time: NPD vs NPD_NM
    plotLists([ssimListNPD, ssimListNPD_NM],
                X=[timeListNPD, timeListNPD_NM],
                stopIndices=[dpStopIndexNPD, dpStopIndexNPD_NM],
                labels=["NPD", "NPD without momentum"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='SSIM NPD vs NPD without momentum',
                xlabel='Time (seconds)', ylabel='SSIM',
                saveName='graySSIM_NPDvsNPD_NM_TIME.pdf')

    # Plot RRE vs Iteration: NPD vs PNPD
    plotLists([rreListNPD, rreListPNPD],
                stopIndices=[dpStopIndexNPD, dpStopIndexPNPD],
                labels=["NPD", "PNPD"],
                labelsStop=['Discrepancy principle stop' for _ in range(2)],
                title='RRE NPD vs PNPD',
                xlabel='Iteration', ylabel='RRE',
                saveName='grayRRE_NPDvsPNPD_IT.pdf', semilogy=True)
    
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

    # Plot gamma vs Iteration: gammaFFBS vs gammaNPD vs gammaPNPD
    plotLists([gammaListNPD, gammaFFBSListNPD],
              labels=["$\gamma^{NPD}$", "$\gamma^{FFBS}$"],
                title='$\gamma^{FFBS}$ vs $\gamma^{NPD}$',
                xlabel='Iteration', ylabel='$\gamma$',
                linestyle=['-', '-.'],
                saveName='grayGamma_IT.pdf')

    if show:
        plt.show()
