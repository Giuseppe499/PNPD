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
from numpy.linalg import norm
from mathExtras import (sInner, gradLeastSquares, grad2D, div2D, proxhsTV,
                        fftConvolve2D)
from solvers import NPD
import grayConfig

def main():
    with np.load(f'./npz/{grayConfig.prefix}/Blurred.npz') as data:
        b = data['b']
        bFFT = data['bFFT']
        psf = data['psf']
        psfFFT = data['psfFFT']
        psfFFTC = data['psfFFTC']
        image = data['image']
        noiseNormSqd = data['noiseNormSqd']

    maxIt = 80  # Maximum number of iterations
    tol = noiseNormSqd  # Tolerance
    lamValues = grayConfig.lamValues  # TV regularization parameter
    pStep = 1  # Primal step length
    dStep = .99 / 8  # Dual step length
    dp = 1.01 # Discrepancy principle parameter
    kMaxValues =  grayConfig.kMaxValues # Number of dual iterations



    gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)
    mulW=grad2D
    mulWT=div2D
    f=lambda x: sInner((fftConvolve2D(x, psf) - b).ravel())
    rho = lambda i: 1 / (i + 1) ** 1.1

    ################################################################################
    # NPD kMax
    x1_K, imRecNPD_K, rreListNPD_K, ssimListNPD_K, timeListNPD_K, dpStopIndexNPD_K = [], [], [], [], [], []
    for lam in lamValues:
        proxhs=lambda alpha, x: proxhsTV(lam, x)
        x1_Kl, imRecNPD_Kl, rreListNPD_Kl, ssimListNPD_Kl, timeListNPD_Kl, dpStopIndexNPD_Kl = [], [], [], [], [], []
        for kMax in kMaxValues:
            print("NPD kMax = ", kMax)
            x1,imRecNPD, rreListNPD, ssimListNPD, timeListNPD, gammaListNPD, gammaFFBSListNPD, dpStopIndexNPD = \
            NPD(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT, f=f, pStep=pStep, dStep=dStep, kMax=kMax, rho=rho, maxit=maxIt, tol=tol, dp=dp, xOrig=image)
            x1_Kl.append(x1)
            imRecNPD_Kl.append(imRecNPD)
            rreListNPD_Kl.append(rreListNPD)
            ssimListNPD_Kl.append(ssimListNPD)
            timeListNPD_Kl.append(timeListNPD)
            dpStopIndexNPD_Kl.append(dpStopIndexNPD)
            print("\n\n\n\n")
        x1_K.append(x1_Kl)
        imRecNPD_K.append(imRecNPD_Kl)
        rreListNPD_K.append(rreListNPD_Kl)
        ssimListNPD_K.append(ssimListNPD_Kl)
        timeListNPD_K.append(timeListNPD_Kl)
        dpStopIndexNPD_K.append(dpStopIndexNPD_Kl)
    np.savez(f"./npz/{grayConfig.prefix}/NPD_K.npz", x1_K=x1_K, imRecNPD_K=imRecNPD_K, rreListNPD_K=rreListNPD_K, ssimListNPD_K=ssimListNPD_K, timeListNPD_K=timeListNPD_K, dpStopIndexNPD_K=dpStopIndexNPD_K, kMaxValues=kMaxValues, lamValues=lamValues)

if __name__ == "__main__":
    main()