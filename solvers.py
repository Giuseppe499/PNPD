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

def NPD_step(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
             mulWT: Callable, pStep: float, dStep: float, t0: float, C: float,
             rho_i: float, kMax: int = 1):
    t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
    gamma = (t0 - 1) / t
    gamma = min(gamma, C*rho_i / (norm(x1 - x0)))
    xBar = x1 + gamma * (x1 - x0)
    # Primal Dual Iteration
    #k = 0
    x2 = xBar - pStep * gradf(xBar) - pStep * mulWT(y0)
    y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
    y0 = y1
    x1Sum = np.zeros(x1.shape)
    for k in range(1,kMax):
        x2 = xBar - pStep * gradf(xBar) - pStep * mulWT(y0)
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = xBar - pStep * gradf(xBar) - pStep * mulWT(y0)
    x1Sum += x2
    x2 = x1Sum / kMax
    return x1, x2, t, y1

def NPD(x0, gradf: Callable, proxhs: Callable, mulW: Callable, mulWT: Callable,
        f: Callable, h: Callable, pStep: float, dStep: float, xOrig,
        kMax: int = 1, t0: float = 0, rho: Callable = lambda i: 1/(i+1)**1.1,
        tol: float = 1e-4, dp: float = 1, maxit: int = 100):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    rreList = []
    y0 = np.zeros(proxhs(1, mulW(x0)).shape)

    i = 0
    x0, x1, t0, y0 = NPD_step(x0=x0, x1=x0, y0=y0, gradf=gradf, proxhs=proxhs,
                                  mulW=mulW, mulWT=mulWT, pStep=pStep, dStep=dStep,
                                  t0=t0, C=0, rho_i=1, kMax=kMax)
    rre = norm(xOrig - x1) / norm(xOrig)
    rreList.append(rre)
    print("Iteration: " + str(i), end="")
    print(", RRE: " + str(rre))
    if f(x1) < dp*tol:
        print("tol reached")
    else:
        C = 10 * norm(x1 - x0) 
        for i in range(1,maxit):
            x0, x1, t0, y0 = NPD_step(x0=x0, x1=x1, y0=y0, gradf=gradf, proxhs=proxhs,
                                    mulW=mulW, mulWT=mulWT, pStep=pStep, dStep=dStep,
                                    t0=t0, C=C, rho_i=rho(i), kMax=kMax)
            val2 = f(x1) + h(mulW(x1))
            rre = norm(xOrig - x1) / norm(xOrig)
            rreList.append(rre)
            print("Iteration: " + str(i), end="")
            print(", RRE: " + str(rre), end="")
            print(", f(x1) + g(x1): " + str(val2))
            if f(x1) < dp*tol:
                print("tol reached")
                break
    return x1, rreList

def PNPD_step(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
               mulWT: Callable, mulPIn: Callable, pStep: float, dStep: float,
               PReg: float, t0: float, C: float, rho_i: float, kMax: int = 1):
    t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
    gamma = (t0 - 1) / t
    gamma = min(gamma, C*rho_i / (norm(x1 - x0)))
    xBar = x1 + gamma * (x1 - x0)
    # Primal Dual Iteration
    #k = 0
    x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
    y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
    y0 = y1
    x1Sum = np.zeros(x1.shape)
    for k in range(1,kMax):
        x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
    x1Sum += x2
    x2 = x1Sum / kMax
    return x1, x2, t, y1


def PNPD(x0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable, f: Callable, h: Callable,
          pStep: float, dStep: float, PReg: float, xOrig, kMax: int = 1,
          t0: float = 0, rho: Callable = lambda i: 1/(i+1)**1.1, tol: float = 1e-4, dp: float = 1, maxit: int = 100):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    rreList = []
    y0 = np.zeros(proxhs(1, mulW(x0)).shape)

    i = 0
    x0, x1, t0, y0 = PNPD_step(x0=x0, x1=x0, y0=y0, gradf=gradf, proxhs=proxhs,
                                  mulW=mulW, mulWT=mulWT, mulPIn=mulPIn, pStep=pStep,
                                  dStep=dStep, PReg=PReg, t0=t0, C=0, rho_i=1,
                                  kMax=kMax)
    rre = norm(xOrig - x1) / norm(xOrig)
    rreList.append(rre)
    print("Iteration: " + str(i), end="")
    print(", RRE: " + str(rre))
    if f(x1) < dp*tol:
        print("tol reached")
    else:
        C = 10 * norm(x1 - x0) 
        for i in range(1,maxit):
            x0, x1, t0, y0 = PNPD_step(x0=x0, x1=x1, y0=y0, gradf=gradf, proxhs=proxhs,
                                  mulW=mulW, mulWT=mulWT, mulPIn=mulPIn, pStep=pStep,
                                  dStep=dStep, PReg=PReg, t0=t0, C=C, rho_i=rho(i),
                                  kMax=kMax)
            val2 = f(x1) + h(mulW(x1))
            rre = norm(xOrig - x1) / norm(xOrig)
            rreList.append(rre)
            print("Iteration: " + str(i), end="")
            print(", RRE: " + str(rre), end="")
            print(", f(x1) + g(x1): " + str(val2))
            if f(x1) < dp*tol:
                print("tol reached")
                break
    return x1, rreList

import torch
def torch_PNPD_step(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable,
          pStep: float, dStep: float, PReg: float, kMax: int = 1,
          t0: float = 0):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
    xBar = x1 + (t0 - 1) / t * (x1 - x0)
    # Primal Dual Iteration
    x1Sum = torch.zeros(x1.shape, device=x1.device, dtype=x1.dtype)
    for k in range(kMax):
        x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = xBar - pStep * mulPIn(PReg, gradf(xBar)) - pStep * mulWT(y0)
    x1Sum += x2
    x2 = x1Sum / (kMax + 1)
    return x1, x2, t, y1
