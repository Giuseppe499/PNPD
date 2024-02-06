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
from numpy.linalg import norm
from torch.fft import fft2
from torchvision.transforms import ToTensor
import numpy as np
from torchExtras import (gradLeastSquares, grad2D, div2D, proxhsTV, mulPInLeastSquares)
from solvers import torch_PNPD_step
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    with np.load('rgbBlurred.npz') as data:
        b = ToTensor()(data['b'])
        psf = ToTensor()(data['psf'])
        image = ToTensor()(data['image'])

    lam = torch.tensor(1e-2, requires_grad=True).unsqueeze(-1).unsqueeze(-1) # TV regularization parameter
    pStep = 1.  # Primal step length
    dStep = torch.tensor(1/8, requires_grad=True).unsqueeze(-1).unsqueeze(-1)  # Dual step length
    PReg = torch.tensor(1e-0, requires_grad=True).unsqueeze(-1).unsqueeze(-1)  # Parameter for the preconditioner P
    kMax = 5 # Number of dual iterations
    steps = 4

    # Compute FFT of b and psf
    bFFT = fft2(b)
    psfFFT = fft2(psf)
    psfFFTC=torch.conj(psfFFT)
    psfAbsSq = psfFFTC * psfFFT

    # Work on single color channel
    rreList = []
    ssimList = []
    xx0 = b
    xx1 = xx0
    yy0 = torch.zeros(3, 2, 256, 256)
    tt0 = 0
    proxhs=lambda alpha, x: proxhsTV(lam, x)
    mulW=grad2D
    mulWT=div2D
    mulPIn=lambda mu, x: mulPInLeastSquares(mu, x,psfAbsSq[0,:,:])
    for j in range(steps):
        for i in range(3):
            gradf=lambda x: gradLeastSquares(x, bFFT[i,:,:], psfFFT[0,:,:], psfFFTC[0,:,:])
            xx0[i, :, :], xx1[i, :, :], tt0, yy0[i, :, :, :] = torch_PNPD_step(x0=xx0[i, :, :], x1=xx1[i, :, :], y0=yy0[i, :, :, :],gradf=gradf,proxhs=proxhs, mulW=mulW, mulWT=mulWT, mulPIn=mulPIn,
                         pStep=pStep, dStep=dStep, PReg=PReg, t0=tt0, kMax=kMax)
            rreList.append(norm(xx1.detach() - image) / norm(image))
        ssimList.append(ssim(xx1.permute(1, 2, 0).detach().numpy(), image.permute(1, 2, 0).numpy(), multichannel=True, channel_axis=2, data_range=1))
        print(f"xx1: RRE: {rreList[-1]}, SSIM: {ssimList[-1]}")
    xx1 = xx1.unsqueeze(0)

    # Clone psf on the 3 channels for easy computation
    psfFFT = psfFFT.repeat(3, 1, 1)
    psfFFTC = psfFFTC.repeat(3, 1, 1)
    psfAbsSq = psfAbsSq.repeat(3, 1, 1)

    # unsqueeze to simulate batch dimension
    b = b.unsqueeze(0)
    bFFT = bFFT.unsqueeze(0)
    psfFFT = psfFFT.unsqueeze(0)
    psfFFTC = psfFFTC.unsqueeze(0)
    psfAbsSq = psfAbsSq.unsqueeze(0)

    batchSize = 1
    b = b.repeat(batchSize, 1, 1, 1)
    bFFT = bFFT.repeat(batchSize, 1, 1, 1)
    psfFFT = psfFFT.repeat(batchSize, 1, 1, 1)
    psfFFTC = psfFFTC.repeat(batchSize, 1, 1, 1)
    psfAbsSq = psfAbsSq.repeat(batchSize, 1, 1, 1)

    gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)
    proxhs=lambda alpha, x: proxhsTV(lam, x)
    mulW=grad2D
    mulWT=div2D
    mulPIn=lambda mu, x: mulPInLeastSquares(mu, x,psfAbsSq)
    
    x0 = b
    rreList = []
    ssimList = []
    x1 = x0
    y0 = torch.zeros(2, batchSize, 3, 256, 256)
    t0 = 0
    for i in range(steps):
        x0, x1, t0, y0 = torch_PNPD_step(x0=x0, x1=x1, y0=y0,gradf=gradf,proxhs=proxhs, mulW=mulW, mulWT=mulWT, mulPIn=mulPIn,
                         pStep=pStep, dStep=dStep, PReg=PReg, t0=t0, kMax=kMax)
        rreList.append(norm(x1.detach() - image) / norm(image))
        ssimList.append(ssim(x1[0,...].squeeze().permute(1, 2, 0).detach().numpy(), image.permute(1, 2, 0).numpy(), multichannel=True, channel_axis=2, data_range=1))
        print(f"RRE: {rreList[-1]}, SSIM: {ssimList[-1]}")
    x1.sum().backward()

    npx1 = np.load("./rgbPNPD.npz")["imRec"]
    
    print(f"xx1-x1: {xx1-x1}")
    # print(f"xx1-npx1: {xx1.squeeze().permute(1,2,0).detach().numpy() -npx1}")
    # print(f"x1-npx1: {x1.squeeze().permute(1,2,0).detach().numpy()-npx1}")


    # Plot results
    from matplotlib import pyplot as plt

    # plt.plot(rreList)

    # plt.figure()
    # plt.plot(ssimList)
    # plt.show()

    plt.figure()
    plt.imshow(x1[0,...].squeeze().permute(1, 2, 0).detach().numpy())
    plt.title("x1")

    plt.figure()
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title("image")

    # plt.figure()
    # plt.imshow(b.squeeze().permute(1, 2, 0).detach().numpy())
    # plt.title("b")

    plt.figure()
    plt.imshow(xx1[0,...].permute(1, 2, 0).detach().numpy())
    plt.title("xx1")

    # plt.figure()
    # plt.imshow(npx1)
    # plt.title("npx1")

    plt.show()
    