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

__author__ = "Giuseppe Scarlato"
__contact__ = "giuseppe499[at]live.it"
__copyright__ = "Copyright 2023, Giuseppe Scarlato"
__date__ = "2023/10/06"
__license__ = "GPLv3"

from PIL import Image
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2
from mathExtras import (sInner, softTreshold, gradLeastSquares,
                        grad2D, div2D, proxhsTV, fftConvolve2D,
                        generatePsfMatrix, mulPInLeastSquares)
from solvers import deConvolve2D, deConvolve2DThikonov, FISTA, NPD, NPDIT
from matplotlib import pyplot as plt

IMGPATH = "cameraman.tif"

np.random.seed = 42

if __name__ == '__main__':
    # Load image
    image = Image.open(IMGPATH)
image = np.asarray(image) / 255
image = image[::2, ::2]
n = image.shape[0]
print(image.shape)

# Generate PSF
psf = generatePsfMatrix(n, 4)
# plt.imshow(psf)
# plt.show()
# Center PSF
psf = np.roll(psf, (-psf.shape[0] // 2, -psf.shape[0] // 2), axis=(0, 1))
# Generate noise
noise = np.random.normal(size=image.shape)
noise *= 0.01 * norm(image) / norm(noise)
# Generate blurred image
conv = fftConvolve2D(image, psf)

# Add noise to blurred image
b = np.clip(conv + noise, 0, 1)
# Simple treshold deblur
imRec = deConvolve2D(b, psf, 5e-2)
# Thikonov deblur
imRecThikonov = deConvolve2DThikonov(b, psf, 5e-4)

# FISTA deblur
bFFT = fft2(b)
psfFFT = fft2(psf)
psfFFTC = np.conjugate(psfFFT)
psfAbsSq = psfFFTC * psfFFT
imRecNorm1, rreListNorm1 = FISTA(x0=b, tau=5e-4,
                                 gradf=lambda x: gradLeastSquares(x, bFFT,
                                                                  psfFFT,
                                                                  psfFFTC),
                                 f=lambda x: sInner(
                                     (fftConvolve2D(x, psf) - b).ravel()),
                                 stepSize=1, maxit=int(2e2), tol=1e-4,
                                 xOrig=image)

lam = 5e-5
imRecTV, rreListTV = NPD(x0=b,
                         gradf=lambda x: gradLeastSquares(x, bFFT,
                                                          psfFFT,
                                                          psfFFTC),
                         proxhs=lambda alpha, x: proxhsTV(lam, x),
                         mulW=grad2D,
                         mulWT=div2D,
                         f=lambda x: sInner(
                             (fftConvolve2D(x, psf) - b).ravel()),
                         h=lambda y: lam * norm(y.ravel(), 1),
                         pStep=1, dStep=1 / 8, maxit=int(2e2), tol=1e-4,
                         xOrig=image)

imRecTVNPDIT, rreListTVNPDIT = NPDIT(x0=b,
                                     gradf=lambda x: gradLeastSquares(x, bFFT,
                                                                      psfFFT,
                                                                      psfFFTC),
                                     proxhs=lambda alpha, x: proxhsTV(lam, x),
                                     mulW=grad2D,
                                     mulWT=div2D,
                                     mulPIn=lambda mu, x: mulPInLeastSquares(
                                         mu, x, psfAbsSq),
                                     f=lambda x: sInner(
                                         (fftConvolve2D(x, psf) - b).ravel()),
                                     h=lambda y: lam * norm(y.ravel(), 1),
                                     pStep=1, dStep=1 / 8, PReg=1e-1,
                                     maxit=int(2e2), tol=1e-4, xOrig=image)

# Show PSF
# plt.imshow(psf)

# Show blurred image, noise and the sum of these
# f, axs = plt.subplots(1, 3)
# axs[0].imshow(conv, cmap="gray", vmin=0, vmax=1)
# axs[1].imshow(noise, cmap="gray")
# axs[2].imshow(b, cmap="gray", vmin=0, vmax=1)

# Show original image and blurred image
f, axs = plt.subplots(1, 2)
axs[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axs[1].imshow(b, cmap="gray", vmin=0, vmax=1)

# Show results of deblurring methods
f, axs = plt.subplots(2, 2)
axs[0][0].imshow(image, cmap="gray", vmin=0, vmax=1)
axs[0][1].imshow(imRec, cmap="gray", vmin=0, vmax=1)
axs[1][0].imshow(imRecThikonov, cmap="gray", vmin=0, vmax=1)
axs[1][1].imshow(imRecNorm1, cmap="gray", vmin=0, vmax=1)

f, axs = plt.subplots(2, 2)
axs[0][0].imshow(imRecTV, cmap="gray", vmin=0, vmax=1)
axs[0][1].imshow(imRecTVNPDIT, cmap="gray", vmin=0, vmax=1)

# Show RRE for FISTA
plt.figure()
plt.plot(rreListTV)
plt.plot(rreListTVNPDIT)

plt.show()
