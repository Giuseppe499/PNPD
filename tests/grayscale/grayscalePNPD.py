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

maxIt = 150 # Maximum number of iterations
tol = noiseNormSqd # Tolerance
lam = 1e-2 # TV regularization parameter
pStep = 1  # Primal step length
dStep = 1 / 8  # Dual step length
PReg = 1e-1  # Parameter for the preconditioner P
dp = 1.02 # Discrepancy principle parameter
kMax = 1 # Number of dual iterations

gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)
proxhs=lambda alpha, x: proxhsTV(lam, x)
mulW=grad2D
mulWT=div2D
mulPIn=lambda mu, x: mulPInLeastSquares(mu, x, psfAbsSq)
f=lambda x: sInner(fftConvolve2D(x, psf) - b)
rho = lambda i: 1 / (i + 1) ** 1.1

################################################################################
# PNPD  
print("PNPD")
x1,imRecPNPD, rreListPNPD, ssimListPNPD, timeListPNPD, gammaListPNPD, gammaFFBSListPNPD, dpStopIndexPNPD\
        = PNPD(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT,
            mulPIn=mulPIn, f=f, pStep=pStep, dStep=dStep, PReg=PReg,
            dp=dp, maxit=maxIt, tol=tol, xOrig=image, kMax=kMax)
print("\n\n\n\n")

################################################################################
# PNPD without momentum
print("PNPD without momentum")
x1,imRecPNPD_NM, rreListPNPD_NM, ssimListPNPD_NM, timeListPNPD_NM, gammaListPNPD_NM, gammaFFBSListPNPD_NM, dpStopIndexPNPD_NM\
        = PNPD(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT,
            mulPIn=mulPIn, f=f, pStep=pStep, dStep=dStep, PReg=PReg,
            dp=dp, maxit=maxIt, tol=tol, xOrig=image, kMax=kMax, momentum=False)

np.savez("./grayscalePNPD.npz", imRecPNPD=imRecPNPD, rreListPNPD=rreListPNPD, ssimListPNPD=ssimListPNPD, timeListPNPD=timeListPNPD, dpStopIndexPNPD=dpStopIndexPNPD, gammaListPNPD=gammaListPNPD, gammaFFBSListPNPD=gammaFFBSListPNPD,\
            imRecPNPD_NM=imRecPNPD_NM, rreListPNPD_NM=rreListPNPD_NM, ssimListPNPD_NM=ssimListPNPD_NM, timeListPNPD_NM=timeListPNPD_NM, dpStopIndexPNPD_NM=dpStopIndexPNPD_NM, gammaListPNPD_NM=gammaListPNPD_NM, gammaFFBSListPNPD_NM=gammaFFBSListPNPD_NM)
