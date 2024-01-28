import torch
from numpy.linalg import norm
from torch.fft import fft2, ifft2
from torchvision.transforms import ToTensor
import numpy as np
from torchExtras import (gradLeastSquares, grad2D, div2D, proxhsTV, mulPInLeastSquares)
from mathExtras import generatePsfMatrix
from solvers import torch_PNPD_step
from skimage.metrics import structural_similarity as ssim

imageR = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]])
imageG = torch.tensor([[1.8, 1.9, 2.0, 2.1], [2.2, 2.3, 2.4, 2.5], [2.6, 2.7, 2.8, 2.9], [3.0, 3.1, 3.2, 3.3]])
imageB = torch.tensor([[3.4, 3.5, 3.6, 3.7], [3.8, 3.9, 4.0, 4.1], [4.2, 4.3, 4.4, 4.5], [4.6, 4.7, 4.8, 4.9]])

def toRGB(imageR):
    return torch.stack((imageR, imageG, imageB))

def toGrayscale(image, channel=0):
    return image[channel, :, :]

imageRGB = toRGB(imageR)

assert torch.all(toGrayscale(imageRGB) == imageR)

g = grad2D(imageR)
gRGB = grad2D(imageRGB)

assert torch.all(toGrayscale(gRGB[0, :, :]) == g[0, :, :])
assert torch.all(toGrayscale(gRGB[1, :, :]) == g[1, :, :])

d = div2D(g)
dRGB = div2D(gRGB)

assert torch.all(toGrayscale(dRGB) == d)

lam = torch.tensor(1e-1, requires_grad=True).unsqueeze(-1).unsqueeze(-1)
proxhs=lambda alpha, x: proxhsTV(lam, x)

p = proxhs(1, g)
pRGB = proxhs(1, gRGB)

assert torch.all(toGrayscale(pRGB[0, :, :]) == p[0, :, :])
assert torch.all(toGrayscale(pRGB[1, :, :]) == p[1, :, :])

# Generate PSF
psf = generatePsfMatrix(4, 1.6)
# Center PSF
psf = np.roll(psf, (-psf.shape[0] // 2, -psf.shape[0] // 2), axis=(0, 1))
psf = torch.tensor(psf)

# Compute FFT of b and psf
bFFT = fft2(imageR)
psfFFT = fft2(psf)
psfFFTC=torch.conj(psfFFT)
psfAbsSq = psfFFTC * psfFFT

gradf=lambda x: gradLeastSquares(x, bFFT, psfFFT, psfFFTC)

grad = gradf(imageR)

psfRGB = toRGB(psf)
bFFTRGB = fft2(imageRGB)
psfFFTRGB = fft2(psfRGB)
psfFFTCRGB = torch.conj(psfFFTRGB)
psfAbsSqRGB = psfFFTCRGB * psfFFTRGB

assert torch.all(toGrayscale(bFFTRGB) == bFFT)
assert torch.all(toGrayscale(psfFFTRGB) == psfFFT)
assert torch.all(toGrayscale(psfFFTCRGB) == psfFFTC)
assert torch.all(toGrayscale(psfAbsSqRGB) == psfAbsSq)

gradfRGB=lambda x: gradLeastSquares(x, bFFTRGB, psfFFTRGB, psfFFTCRGB)

gradRGB = gradfRGB(imageRGB)

assert torch.all(toGrayscale(gradRGB) == grad)

mu = 1e-1

mulPIn=lambda mu, x: mulPInLeastSquares(mu, x, psfAbsSq)

timesP = mulPIn(mu, imageR)

mulPInRGB=lambda mu, x: mulPInLeastSquares(mu, x, psfAbsSqRGB)

timesPRGB = mulPInRGB(mu, imageRGB)

assert torch.all(toGrayscale(timesPRGB) == timesP)




