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
                        fftConvolve2D)
from solvers import NPD

with np.load('grayscaleBlurred.npz') as data:
    b = data['b']
    bFFT = data['bFFT']
    psf = data['psf']
    psfFFT = data['psfFFT']
    psfFFTC = data['psfFFTC']
    image = data['image']
    noiseNormSqd = data['noiseNormSqd']

maxIt = 150  # Maximum number of iterations
tol = noiseNormSqd  # Tolerance
lam = 1e-3  # TV regularization parameter
pStep = 1  # Primal step length
dStep = 1 / 8  # Dual step length
dp = 1.02 # Discrepancy principle parameter
kMax = 1 # Number of dual iterations



gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)
proxhs=lambda alpha, x: proxhsTV(lam, x)
mulW=grad2D
mulWT=div2D
f=lambda x: sInner(fftConvolve2D(x, psf) - b)
rho = lambda i: 1 / (i + 1) ** 1.1

################################################################################
# NPD
print("NPD")
x1,imRecNPD, rreListNPD, ssimListNPD, timeListNPD, gammaListNPD, gammaFFBSListNPD, dpStopIndexNPD = \
NPD(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT, f=f, pStep=pStep, dStep=dStep, kMax=kMax, rho=rho, maxit=maxIt, tol=tol, dp=dp, xOrig=image)
print("\n\n\n\n")

################################################################################
# NPD without momentum
print("NPD without momentum")
x1,imRecNPD_NM, rreListNPD_NM, ssimListNPD_NM, timeListNPD_NM, gammaListNPD_NM, gammaFFBSListNPD_NM, dpStopIndexNPD_NM = \
NPD(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT, f=f, pStep=pStep, dStep=dStep, kMax=kMax, rho=rho, maxit=maxIt, tol=tol, dp=dp, xOrig=image, momentum=False)
print("\n\n\n\n")

np.savez("./grayscaleNPD.npz", imRecNPD=imRecNPD, rreListNPD=rreListNPD, ssimListNPD=ssimListNPD, timeListNPD=timeListNPD, dpStopIndexNPD=dpStopIndexNPD, gammaListNPD=gammaListNPD, gammaFFBSListNPD=gammaFFBSListNPD,\
            imRecNPD_NM=imRecNPD_NM, rreListNPD_NM=rreListNPD_NM, ssimListNPD_NM=ssimListNPD_NM, timeListNPD_NM=timeListNPD_NM, dpStopIndexNPD_NM=dpStopIndexNPD_NM, gammaListNPD_NM=gammaListNPD_NM, gammaFFBSListNPD_NM=gammaFFBSListNPD_NM)
