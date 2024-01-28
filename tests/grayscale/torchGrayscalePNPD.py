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

import torch
from torch.fft import fft2
from torchvision.transforms import ToTensor
import numpy as np
from numpy.linalg import norm

from torchExtras import (gradLeastSquares, grad2D, div2D, proxhsTV, mulPInLeastSquares)
from solvers import torch_PNPD_step

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    with np.load('grayscaleBlurred.npz') as data:
        b = ToTensor()(data['b'])
        psf = ToTensor()(data['psf'])
        image = ToTensor()(data['image'])

    lam = torch.tensor(5e-7, requires_grad=True) # TV regularization parameter
    pStep = 1  # Primal step length
    dStep = torch.tensor(1 / 8, requires_grad=True)  # Dual step length
    PReg = torch.tensor(1e-1, requires_grad=True)  # Parameter for the preconditioner P

    bFFT = fft2(b)
    psfFFT = fft2(psf)
    psfFFTC = torch.conj(psfFFT)
    psfAbsSq = psfFFTC * psfFFT

    gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT,psfFFTC)
    proxhs=lambda alpha, x: proxhsTV(lam, x)
    mulW=grad2D
    mulWT=div2D
    mulPIn=lambda mu, x: mulPInLeastSquares(mu, x, psfAbsSq)

    x0 = b
    rreList = []
    x1 = x0
    y0 = proxhs(dStep / pStep, dStep / pStep * mulW(x1))
    t0 = 0
    for i in range(10):
        x0, x1, t0, y0 = torch_PNPD_step(x0=x0, x1=x1, y0=y0,gradf=gradf,proxhs=proxhs, mulW=mulW, mulWT=mulWT, mulPIn=mulPIn,                           
                           pStep=pStep, dStep=dStep, PReg=PReg, t0=t0)
        rreList.append(norm(x1.detach()-image) / norm(image))
        print(f"RRE: {rreList[-1]}")
    x1.sum().backward()
    print(lam.grad)
    print(dStep.grad)
    print(PReg.grad)
