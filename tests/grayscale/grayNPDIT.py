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
                        fftConvolve2D, mulPInLeastSquares, mulPLeastSquares)
from solvers import NPDIT
import grayConfig

def main():
    with np.load(f'./npz/{grayConfig.prefix}/Blurred.npz') as data:
        b = data['b']
        bFFT = data['bFFT']
        psf = data['psf']
        psfFFT = data['psfFFT']
        psfFFTC = data['psfFFTC']
        psfAbsSq = data['psfAbsSq']
        image = data['image']
        noiseNormSqd = data['noiseNormSqd']

    maxIt = grayConfig.maxIt # Maximum number of iterations
    tol = noiseNormSqd # Tolerance
    lam = grayConfig.lam # TV regularization parameter
    L = 0.1  # Estimate of the Lipschitz constant of the gradient of f
    normWsqrd = 8 # Estimate of the norm squared of the operator W
    PReg = grayConfig.nu  # Parameter for the preconditioner P
    dp = 1.02 # Discrepancy principle parameter
    kMax = grayConfig.kMax # Number of dual iterations

    gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)
    proxhs=lambda alpha, x: proxhsTV(lam, x)
    mulW=grad2D
    mulWT=div2D
    mulPIn=lambda mu, x: mulPInLeastSquares(mu, x, psfAbsSq)
    mulP=lambda mu, x: mulPLeastSquares(mu, x, psfAbsSq)
    f=lambda x: sInner(fftConvolve2D(x, psf) - b)
    rho = lambda i: 1 / (i + 1) ** 1.1

    ################################################################################
    # NPDIT  
    print("NPDIT")
    x1,imRec, rreList, ssimList, timeList, gammaList, gammaFFBSList, dpStopIndex, recList\
            = NPDIT(x0=b, gradf=gradf, proxhs=proxhs, mulW=mulW, mulWT=mulWT,
                mulPIn=mulPIn, mulP=mulP, f=f, L=L, normWsqrd=normWsqrd, PReg=PReg, rho=rho,
                dp=dp, maxit=maxIt, tol=tol, xOrig=image, kMax=kMax, momentum=grayConfig.momentum, recIndexes=grayConfig.recIndexes)
    print("\n\n\n\n")

    np.savez(f"./npz/{grayConfig.prefix}/NPDIT_{grayConfig.suffix}.npz", imRec=imRec, rreList=rreList, ssimList=ssimList, timeList=timeList, dpStopIndex=dpStopIndex, gammaList=gammaList, gammaFFBSList=gammaFFBSList, recList=recList)

if __name__ == "__main__":
    main()