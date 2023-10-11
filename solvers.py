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
from mathExtras import sInner


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
    val0 = f(x0) + g(x0)
    for i in range(maxit):
        x1 = proxg(stepSize, x0 - stepSize * gradf(x0))
        val1 = f(x1) + g(x1)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(norm(xOrig - x1) / norm(xOrig)), end="")
        print(", f(x1) + g(x1): " + str(val1), end="")
        print(", delta val: " + str(val1 - val1))
        if val0 - val1 < tol:
            print(val0 - val1)
            return x1
        xlast = x0
        x0 = x1
    print("Exceded maxit")
    return x1


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
        print(", f(x1) + g(x1): " + str(val1), end="")
        print(", delta val: " + str(val1 - val1))

        if np.abs(val1 - val2) < tol:
            break
        # plt.imshow(x, cmap="gray", vmin=0, vmax=1)
        # plt.draw()
        # plt.pause(0.001)
        x0 = x1
        x1 = x2
        val1 = val2
        t0 = t
    print("Exceded maxit")
    return x2, rreList
