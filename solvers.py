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

from typing import Callable

import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2, ifft2


def deConvolve2D(conv, psf, epsilon: float):
    return np.real(ifft2(fft2(conv) / np.clip(fft2(psf), epsilon, None)))


def deConvolve2DThikonov(conv, psf, alpha):
    psfFFT = fft2(psf)
    psfFFTC = np.conjugate(psfFFT)
    convFFT = fft2(conv)
    return np.real(ifft2(psfFFTC * convFFT / (psfFFTC * psfFFT + alpha)))

def deConvolve2DThikonovPlusEstimate(conv, psf, estimate, alpha):
    psfFFT = fft2(psf)
    psfFFTC = np.conjugate(psfFFT)
    estimateFFT = fft2(estimate)
    convFFT = fft2(conv)
    return np.real(ifft2((psfFFTC * convFFT + alpha*estimateFFT) / (psfFFTC * psfFFT + alpha)))


def NPD(x0, gradf: Callable, proxhs: Callable, mulW: Callable, mulWT: Callable,
        f: Callable, h: Callable, pStep: float, dStep: float, xOrig,
        kMax: int = 1, t0: float = 0, tol: float = 1e-4, dp: float = 1, maxit: int = 100):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    rreList = []
    x1 = x0
    val1 = f(x1) + h(mulW(x1))
    y0 = proxhs(dStep / pStep, dStep / pStep * mulW(x1))
    for i in range(maxit):
        t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
        xBar = x1 + (t0 - 1) / t * (x1 - x0)
        # Primal Dual Iteration
        x1Sum = np.zeros(x1.shape)
        for k in range(kMax):
            x2 = xBar - pStep * gradf(xBar) - pStep * mulWT(y0)
            x1Sum += x2
            y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
            y0 = y1
        x2 = xBar - pStep * gradf(xBar) - pStep * mulWT(y0)
        x1Sum += x2
        x2 = x1Sum / (kMax + 1)
        val2 = f(x2) + h(mulW(x2))
        rre = norm(xOrig - x2) / norm(xOrig)
        rreList.append(rre)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", f(x1) + g(x1): " + str(val2), end="")
        print(", f(x2): " + str(f(x2)))
        if f(x2) < dp*tol:
            print("tol reached")
            break
        x0 = x1
        x1 = x2
        val1 = val2
        t0 = t
    return x2, rreList


def NPDIT(x0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable, f: Callable, h: Callable,
          pStep: float, dStep: float, PReg: float, xOrig, kMax: int = 1,
          t0: float = 0, tol: float = 1e-4, dp: float = 1, maxit: int = 100):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    rreList = []
    x1 = x0
    val1 = f(x1) + h(mulW(x1))
    y0 = proxhs(dStep / pStep, dStep / pStep * mulW(x1))
    for i in range(maxit):
        t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
        xBar = x1 + (t0 - 1) / t * (x1 - x0)
        # Primal Dual Iteration
        x1Sum = np.zeros(x1.shape)
        for k in range(kMax):
            x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
            x1Sum += x2
            y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
            y0 = y1
        x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
        x1Sum += x2
        x2 = x1Sum / (kMax + 1)
        val2 = f(x2) + h(mulW(x2))
        rre = norm(xOrig - x2) / norm(xOrig)
        rreList.append(rre)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", f(x1) + g(x1): " + str(val2), end="")
        print(", f(x2): " + str(f(x2)))
        if f(x2) < dp*tol:
            print("tol reached")
            break
        x0 = x1
        x1 = x2
        val1 = val2
        t0 = t
    return x2, rreList

import torch
def torch_NPDIT_step(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable,
          pStep: float, dStep: float, PReg: float, xOrig, kMax: int = 1,
          t0: float = 0):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
    xBar = x1 + (t0 - 1) / t * (x1 - x0)
    # Primal Dual Iteration
    x1Sum = torch.zeros(x1.shape)
    for k in range(kMax):
        x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
    x1Sum += x2
    x2 = x1Sum / (kMax + 1)
    # Activate only for debugging
    # rre = norm(xOrig.detach() - x2.detach()) / norm(xOrig.detach())
    # print("RRE: " + str(rre))
    return x1, x2, t
