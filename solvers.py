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
        stepSize: float, tol: float = 1e-4, maxit: int = 100):
    """
    Forward-Backward Splitting
    Approximate argmin_{x \in R^d} f(x) + g(x) where f is differentiable and
    the proximity operator of g is known.
    """
    for i in range(maxit):
        x = proxg(stepSize, x0 - stepSize * gradf(x0))
        print(norm(x - x0))
        if norm(x - x0) / norm(x) < tol:
            print(i)
            print(norm(x - x0))
            return x
        xlast = x0
        x0 = x
    print(norm(x - xlast))
    print("Exceded maxit")
    return x


def FBSB(x0, gradf: Callable, proxg: Callable, f: Callable, g: Callable,
         stepSize: float, rho: float = .5, tol: float = 1e-4, maxit: int = 100):
    """
    Forward-Backward Splitting with Backtracking
    Approximate argmin_{x \in R^d} f(x) + g(x) where f is differentiable and
    the proximity operator of g is known.
    """
    val0 = f(x0) + g(x0)
    gradfx0 = gradf(x0);
    for i in range(maxit):
        stepSize = stepSize
        x = proxg(stepSize, x0 - stepSize * gradfx0)
        val = f(x) + g(x)
        while val > val0 - 1e-4 * stepSize * sInner(gradfx0.ravel()):
            stepSize = rho * stepSize
            x = proxg(stepSize, x0 - stepSize * gradfx0)
            val = f(x) + g(x)
            if stepSize < 1e-16:
                print("stepSize too short")
                return x
        print(stepSize)
        print(val)
        if val0 - val < tol:
            print(i)
            print(val - val0)
            return x
        # plt.imshow(x, cmap="gray", vmin=0, vmax=1)
        # plt.draw()
        # plt.pause(0.001)
        x0 = x
        val0 = val
        gradfx0 = gradf(x0);
    print(val)
    print("Exceded maxit")
    return x


def FFBS(x0, gradf: Callable, proxg: Callable, f: Callable, g: Callable,
         stepSize: float, t0: float = 1, tol: float = 1e-4, maxit: int = 100):
    """
    Fast Forward-Backward Splitting (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + g(x) where f is differentiable and
    the proximity operator of g is known.
    """
    xLast = x0
    x0 = proxg(stepSize, xLast - stepSize * gradf(xLast))
    val0 = f(x0) + g(x0)
    for i in range(maxit):
        t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
        y = x0 + (t0 - 1) / t * (x0 - xLast)
        x = proxg(stepSize, y - stepSize * gradf(y))
        val = f(x) + g(x)
        print(val0 - val)
        if np.abs(val0 - val) < tol:
            print(i)
            print(val0 - val)
            return x
        # plt.imshow(x, cmap="gray", vmin=0, vmax=1)
        # plt.draw()
        # plt.pause(0.001)
        xLast = x0
        x0 = x
        val0 = val
        t0 = t
    print(val)
    print("Exceded maxit")
    return x
