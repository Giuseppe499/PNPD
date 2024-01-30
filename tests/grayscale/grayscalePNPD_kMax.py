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

import numpy as np
from numpy.linalg import norm
from mathExtras import (sInner, gradLeastSquares, grad2D, div2D, proxhsTV,
                        fftConvolve2D, mulPInLeastSquares)
from solvers import PNPD

with np.load('grayscaleBlurred.npz') as data:
    b = data['b']
    bFFT = data['bFFT']
    psf = data['psf']
    psfFFT = data['psfFFT']
    psfFFTC = data['psfFFTC']
    psfAbsSq = data['psfAbsSq']
    image = data['image']
    noiseNormSqd = data['noiseNormSqd']

maxIt = 80  # Maximum number of iterations
tol = noiseNormSqd  # Tolerance
lamValues = [1e-3, 1e-2]  # TV regularization parameter
pStep = 1  # Primal step length
dStep = 1 / 8  # Dual step length
PReg = 1e-1  # Parameter for the preconditioner P
dp = 1.01 # Discrepancy principle parameter
kMaxValues = [1,2,5,10] # Number of dual iterations



gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)
mulW=grad2D
mulWT=div2D
mulPIn=lambda mu, x: mulPInLeastSquares(mu, x, psfAbsSq)
f=lambda x: sInner((fftConvolve2D(x, psf) - b).ravel())
rho = lambda i: 1 / (i + 1) ** 1.1

################################################################################
# PNPD kMax
x1_K, imRecPNPD_K, rreListPNPD_K, ssimListPNPD_K, timeListPNPD_K, dpStopIndexPNPD_K = [], [], [], [], [], []
for lam in lamValues:
    proxhs=lambda alpha, x: proxhsTV(lam, x)
    x1_Kl, imRecPNPD_Kl, rreListPNPD_Kl, ssimListPNPD_Kl, timeListPNPD_Kl, dpStopIndexPNPD_Kl = [], [], [], [], [], []
    for kMax in kMaxValues:
        print("PNPD kMax = ", kMax)
        x1,imRecPNPD, rreListPNPD, ssimListPNPD, timeListPNPD, gammaListPNPD, gammaFFBSListPNPD, dpStopIndexPNPD = \
        PNPD(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT,
            mulPIn=mulPIn, f=f, pStep=pStep, dStep=dStep, PReg=PReg,
            dp=dp, maxit=maxIt, tol=tol, xOrig=image, kMax=kMax)
        x1_Kl.append(x1)
        imRecPNPD_Kl.append(imRecPNPD)
        rreListPNPD_Kl.append(rreListPNPD)
        ssimListPNPD_Kl.append(ssimListPNPD)
        timeListPNPD_Kl.append(timeListPNPD)
        dpStopIndexPNPD_Kl.append(dpStopIndexPNPD)
        print("\n\n\n\n")
    x1_K.append(x1_Kl)
    imRecPNPD_K.append(imRecPNPD_Kl)
    rreListPNPD_K.append(rreListPNPD_Kl)
    ssimListPNPD_K.append(ssimListPNPD_Kl)
    timeListPNPD_K.append(timeListPNPD_Kl)
    dpStopIndexPNPD_K.append(dpStopIndexPNPD_Kl)
np.savez(f"./grayscalePNPD_K.npz", x1_K=x1_K, imRecPNPD_K=imRecPNPD_K, rreListPNPD_K=rreListPNPD_K, ssimListPNPD_K=ssimListPNPD_K, timeListPNPD_K=timeListPNPD_K, dpStopIndexPNPD_K=dpStopIndexPNPD_K, kMaxValues=kMaxValues, lamValues=lamValues)
    

