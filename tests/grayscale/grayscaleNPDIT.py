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
from numpy.fft import fft2
from mathExtras import (sInner, gradLeastSquares, grad2D, div2D, proxhsTV,
                        fftConvolve2D, mulPInLeastSquares)
from solvers import NPDIT

if __name__ == '__main__':
    with np.load('grayscaleBlurred.npz') as data:
        b = data['b']
        psf = data['psf']
        image = data['image']
        noiseNormSqd = data['noiseNormSqd']

    maxIt = 50 # Maximum number of iterations
    tol = noiseNormSqd # Tolerance
    lam = 1e-4 # TV regularization parameter
    pStep = 1  # Primal step length
    dStep = 1 / 8  # Dual step length
    PReg = 5e-1  # Parameter for the preconditioner P
    dp = 1 # Discrepancy principle parameter
    kMax = 100 # Number of dual iterations

    bFFT = fft2(b)
    psfFFT = fft2(psf)
    psfFFTC = np.conjugate(psfFFT)
    psfAbsSq = psfFFTC * psfFFT
    imRec, rreList = NPDIT(x0=b,
                           gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT,
                                                            psfFFTC),
                           proxhs=lambda alpha, x: proxhsTV(lam, x),
                           mulW=grad2D,
                           mulWT=div2D,
                           mulPIn=lambda mu, x: mulPInLeastSquares(mu, x,
                                                                   psfAbsSq),
                           f=lambda x: sInner(
                               (fftConvolve2D(x, psf) - b).ravel()),
                           h=lambda y: lam * norm(y.ravel(), 1),
                           pStep=pStep, dStep=dStep, PReg=PReg, dp=dp,
                           maxit=maxIt, tol=tol, xOrig=image, kMax=kMax)

    np.savez("./grayscaleNPDIT.npz", imRec=imRec, rreList=rreList)
