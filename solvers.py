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
from mathExtras import softTreshold


def deConvolve2D(conv, psf, epsilon: float):
    return np.real(ifft2(fft2(conv) / np.clip(fft2(psf), epsilon, None)))


def deConvolve2DThikonov(conv, psf, alpha):
    psfFFT = fft2(psf)
    psfFFTC = np.conjugate(psfFFT)
    convFFT = fft2(conv)
    return np.real(ifft2(psfFFTC * convFFT / (psfFFTC * psfFFT + alpha)))


def FBS(x0, gradf: Callable, proxg: Callable, f: Callable, g: Callable,
        stepSize: float, xOrig, tol: float = 1e-4, maxit: int = 100):
    """
    Forward-Backward Splitting
    Approximate argmin_{x \in R^d} f(x) + g(x) where f is differentiable and
    the proximity operator of g is known.
    """
    rreList = []
    val0 = f(x0) + g(x0)
    for i in range(maxit):
        x1 = proxg(stepSize, x0 - stepSize * gradf(x0))
        val1 = f(x1) + g(x1)
        rre = norm(xOrig - x1) / norm(xOrig)
        rreList.append(rre)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", f(x1) + g(x1): " + str(val1), end="")
        print(", delta val: " + str(val1 - val0))
        if val0 - val1 < tol:
            print("tol reached")
            break
        x0 = x1
    return x1, rreList


def FFBS(x0, gradf: Callable, proxg: Callable, f: Callable, g: Callable,
         stepSize: float, xOrig, t0: float = 1, tol: float = 1e-4, maxit: int
         = 100):
    """
    Fast Forward-Backward Splitting (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + g(x) where f is differentiable and
    the proximity operator of g is known.
    """
    rreList = []
    x1 = proxg(stepSize, x0 - stepSize * gradf(x0))
    val1 = f(x1) + g(x1)
    for i in range(maxit):
        t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
        y = x1 + (t0 - 1) / t * (x1 - x0)
        x2 = proxg(stepSize, y - stepSize * gradf(y))
        val2 = f(x2) + g(x2)
        rre = norm(xOrig - x2)
        rreList.append(rre)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", f(x2) + g(x2): " + str(val2), end="")
        print(", delta val: " + str(val2 - val1))

        if np.abs(val1 - val2) < tol:
            print("tol reached")
            break
        # plt.imshow(x, cmap="gray", vmin=0, vmax=1)
        # plt.draw()
        # plt.pause(0.001)
        x0 = x1
        x1 = x2
        val1 = val2
        t0 = t
    return x2, rreList


def FISTA(x0, gradf: Callable, f: Callable, tau: float, stepSize: float,
          xOrig, t0: float = 1, tol: float = 1e-4, maxit: int = 100):
    """
    Fast Iterative Soft Tresholding Algorithm
    Approximate argmin_{x \in R^d} f(x) + tau*|x|_1 where f is
    differentiable
    """
    return FFBS(x0, gradf=gradf,
                proxg=lambda alpha, x: softTreshold(alpha * tau, x),
                f=f, g=lambda x: tau * norm(x), stepSize=stepSize,
                maxit=maxit, tol=tol, xOrig=xOrig)


def NPD(x0, gradf: Callable, proxhs: Callable, mulW: Callable, mulWT: Callable,
        f: Callable, h: Callable, pStep: float, dStep: float, xOrig,
        kMax: int = 1, t0: float = 0, tol: float = 1e-4, maxit: int = 100):
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
        print(", delta val: " + str(val2 - val1))
        if abs(val1 - val2) < tol:
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
          t0: float = 0, tol: float = 1e-4, maxit: int = 100):
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
        print(", delta val: " + str(val2 - val1))
        if abs(val1 - val2) < tol:
            print("tol reached")
            break
        x0 = x1
        x1 = x2
        val1 = val2
        t0 = t
    return x2, rreList
