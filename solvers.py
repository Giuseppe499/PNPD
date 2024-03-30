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
from skimage.metrics import structural_similarity as ssim
from mathExtras import sInner

import time

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
    gammaFFBS = (t0 - 1) / t
    gamma = min(gammaFFBS, C*rho_i / (norm(x1 - x0)))
    xBar = x1 + gamma * (x1 - x0)
    # Primal Dual Iteration
    #k = 0
    gradf_xBar = gradf(xBar)
    x2 = xBar - pStep * (gradf_xBar + mulWT(y0))
    y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
    y0 = y1
    x1Sum = np.zeros(x1.shape)
    for k in range(1,kMax):
        x2 = xBar - pStep * (gradf_xBar + mulWT(y0))
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = xBar - pStep * (gradf_xBar + mulWT(y0))
    x1Sum += x2
    x2 = x1Sum / kMax
    return x1, x2, t, y1, gamma, gammaFFBS

def NPD_step_no_momentum(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
             mulWT: Callable, pStep: float, dStep: float, t0: float, C: float,
             rho_i: float, kMax: int = 1):
    # Primal Dual Iteration
    #k = 0
    gradf_x1 = gradf(x1)
    x2 = x1 - pStep * (gradf_x1 + mulWT(y0))
    y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
    y0 = y1
    x1Sum = np.zeros(x1.shape)
    for k in range(1,kMax):
        x2 = x1 - pStep * (gradf_x1 + mulWT(y0))
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = x1 - pStep * (gradf_x1 + mulWT(y0))
    x1Sum += x2
    x2 = x1Sum / kMax
    return x1, x2, 0, y1, 0, 0

def NPD(x0, gradf: Callable, proxhs: Callable, mulW: Callable, mulWT: Callable,
        f: Callable, pStep: float, dStep: float, xOrig,
        kMax: int = 1, t0: float = 1, rho: Callable = lambda i: 1/(i+1)**1.1,
        tol: float = 1e-4, dp: float = 1, maxit: int = 100, momentum: bool = True, recIndexes = []):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    dpReached = False
    imRec = None

    timeList = []
    rreList = []
    ssimList = []
    gammaList = []
    gammaFFBSList = []

    recList = []

    x1 = x0
    y0 = np.zeros(proxhs(1, mulW(x0)).shape)

    Step = NPD_step if momentum else NPD_step_no_momentum

    C = 0
    for i in range(maxit):
        start = time.process_time()
        x0, x1, t0, y0, gamma, gammaFFBS = Step(x0=x0, x1=x1, y0=y0, gradf=gradf, proxhs=proxhs,
                                mulW=mulW, mulWT=mulWT, pStep=pStep, dStep=dStep,
                                t0=t0, C=C, rho_i=rho(i), kMax=kMax)
        elapsed = time.process_time() - start
        if i == 0:
            C = 10 * norm(x1 - x0)
        rre = norm(xOrig - x1) / norm(xOrig)
        rreList.append(rre)
        timeList.append(elapsed)
        ssimList.append(ssim(x1, xOrig, data_range=1))
        gammaList.append(gamma)
        gammaFFBSList.append(gammaFFBS)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", SSIM: " + str(ssimList[-1]))
        if f(x1) < dp*tol and not dpReached :
            imRec = x1
            dpStopIndex = i+1
            dpReached = True
        if (i+1) in recIndexes:
            recList.append(x1)
    
    if imRec is None:
        imRec = x1
        dpStopIndex = i+1
    return x1, imRec, rreList, ssimList, timeList, gammaList, gammaFFBSList, dpStopIndex, recList

def PNPD_step(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
               mulWT: Callable, mulPIn: Callable, pStep: float, dStep: float,
               PReg: float, t0: float, C: float, rho_i: float, kMax: int = 1):
    t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
    gammaFFBS = (t0 - 1) / t
    gamma = min(gammaFFBS, C*rho_i / (norm(x1 - x0)))
    xBar = x1 + gamma * (x1 - x0)
    # Primal Dual Iteration
    #k = 0
    mulPinxBar = mulPIn(PReg, gradf(xBar))
    x2 = xBar - pStep * (mulPinxBar + mulWT(y0))
    y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
    y0 = y1
    x1Sum = np.zeros(x1.shape)
    for k in range(1,kMax):
        x2 = xBar - pStep * (mulPinxBar + mulWT(y0))
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = xBar - pStep * (mulPinxBar + mulWT(y0))
    x1Sum += x2
    x2 = x1Sum / kMax
    return x1, x2, t, y1, gamma, gammaFFBS

def PNPD_step_no_momentum(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
               mulWT: Callable, mulPIn: Callable, pStep: float, dStep: float,
               PReg: float, t0: float, C: float, rho_i: float, kMax: int = 1):
    # Primal Dual Iteration
    #k = 0
    mulPinx1 = mulPIn(PReg, gradf(x1))
    x2 = x1 - pStep * (mulPinx1 + mulWT(y0))
    y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
    y0 = y1
    x1Sum = np.zeros(x1.shape)
    for k in range(1,kMax):
        x2 = x1 - pStep * (mulPinx1 + mulWT(y0))
        x1Sum += x2
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
    x2 = x1 - pStep * (mulPinx1 + mulWT(y0))
    x1Sum += x2
    x2 = x1Sum / kMax
    return x1, x2, None, y1, None, None


def PNPD(x0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable, f: Callable,
          pStep: float, dStep: float, PReg: float, xOrig, kMax: int = 1,
          t0: float = 1, rho: Callable = lambda i: 1/(i+1)**1.1, tol: float = 1e-4, dp: float = 1, maxit: int = 100, momentum: bool = True, recIndexes = []):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    dpReached = False
    imRec = None

    timeList = []
    rreList = []
    ssimList = []
    gammaList = []
    gammaFFBSList = []

    recList = []

    x1 = x0
    y0 = np.zeros(proxhs(1, mulW(x0)).shape)

    Step = PNPD_step if momentum else PNPD_step_no_momentum

    C = 0
    for i in range(maxit):
        start = time.process_time()
        x0, x1, t0, y0, gamma, gammaFFBS = Step(x0=x0, x1=x1, y0=y0, gradf=gradf, proxhs=proxhs,
                                mulW=mulW, mulWT=mulWT, mulPIn=mulPIn, pStep=pStep,
                                dStep=dStep, PReg=PReg, t0=t0, C=C, rho_i=rho(i),
                                kMax=kMax)
        elapsed = time.process_time() - start
        if i == 0:
            C = 10 * norm(x1 - x0)
        rre = norm(xOrig - x1) / norm(xOrig)
        rreList.append(rre)
        timeList.append(elapsed)
        ssimList.append(ssim(x1, xOrig, data_range=1))
        gammaList.append(gamma)
        gammaFFBSList.append(gammaFFBS)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", SSIM: " + str(ssimList[-1]))
        if f(x1) < dp*tol and not dpReached :
            imRec = x1
            dpStopIndex = i+1
            dpReached = True
        if (i+1) in recIndexes:
            recList.append(x1)
    if imRec is None:
        imRec = x1
        dpStopIndex = i+1
    return x1, imRec, rreList, ssimList, timeList, gammaList, gammaFFBSList, dpStopIndex, recList

def NPDIT_step(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable, mulP: Callable, f: Callable,
          L: float, dStep: float, PReg: float, t0: float, C: float, rho_i: float, kMax: int, eps: float, dInv):
    t = .5 + .5 * np.sqrt(1 + 4 * t0 * t0)
    gammaFFBS = (t0 - 1) / t
    gamma = min(gammaFFBS, C*rho_i / (norm(x1 - x0)))
    xBar = x1 + gamma * (x1 - x0)
    gradf_xBar = gradf(xBar)
    # Primal Dual Iteration
    #k = 0
    while True:
        pStep = eps / L
        x2 = xBar - pStep * mulPIn(PReg, gradf_xBar + mulWT(y0))
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
        x1Sum = np.zeros(x1.shape)
        for k in range(1,kMax):
            x2 = xBar - pStep * mulPIn(PReg, gradf_xBar + mulWT(y0))
            x1Sum += x2
            y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
            y0 = y1
        x2 = xBar - pStep * mulPIn(PReg, gradf_xBar + mulWT(y0))
        x1Sum += x2
        x2 = x1Sum / kMax
        x2MinusXBar = x2 - xBar
        if f(x2) <= f(xBar) + sInner(gradf_xBar, x2MinusXBar) + L / 2 * sInner(x2MinusXBar, mulP(PReg, x2MinusXBar)):
            break
        L *= dInv
    return x1, x2, t, y1, L, gamma, gammaFFBS

def NPDIT_step_no_momentum(x0, x1, y0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable, mulP: Callable, f: Callable,
          L: float, dStep: float, PReg: float, t0: float, C: float, rho_i: float, kMax: int, eps: float, dInv):
    # Primal Dual Iteration
    #k = 0
    gradf_x1 = gradf(x1)
    while True:
        pStep = eps / L
        x2 = x1 - pStep * mulPIn(PReg, gradf_x1 + mulWT(y0))
        y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
        y0 = y1
        x1Sum = np.zeros(x1.shape)
        for k in range(1,kMax):
            x2 = x1 - pStep * mulPIn(PReg, gradf_x1 + mulWT(y0))
            x1Sum += x2
            y1 = proxhs(dStep / pStep, y0 + (dStep / pStep) * mulW(x2))
            y0 = y1
        x2 = x1 - pStep * mulPIn(PReg, gradf_x1 + mulWT(y0))
        x1Sum += x2
        x2 = x1Sum / kMax
        x2MinusX1 = x2 - x1
        if f(x2) <= f(x1) + sInner(gradf_x1, x2MinusX1) + L / 2 * sInner(x2MinusX1, mulP(PReg, x2MinusX1)):
            break
        L *= dInv
    return x1, x2, None, y1, L, 0, 0

def NPDIT(x0, gradf: Callable, proxhs: Callable, mulW: Callable,
          mulWT: Callable, mulPIn: Callable, mulP: Callable, f: Callable,
          L: float, normWsqrd: float, PReg: float, xOrig, kMax: int = 1,
          t0: float = 1, eps: float = 0.99, dInv = 2,
          rho: Callable = lambda i: 1/(i+1)**1.1, tol: float = 1e-4,
          dp: float = 1, maxit: int = 100, momentum: bool = True, recIndexes = []):
    """
    Nested Primal Dual (FISTA-like algorithm)
    Approximate argmin_{x \in R^d} f(x) + h(Wx) where f is differentiable and
    the proximity operator of h* (Fenchel conjugate of h) is known.
    """
    dpReached = False
    imRec = None

    timeList = []
    rreList = []
    ssimList = []
    gammaList = []
    gammaFFBSList = []

    recList = []

    x1 = x0
    y0 = np.zeros(proxhs(1, mulW(x0)).shape)
    dStep = eps * PReg / normWsqrd

    Step = NPDIT_step if momentum else NPDIT_step_no_momentum

    C = 0
    for i in range(maxit):
        start = time.process_time()
        x0, x1, t0, y0, L, gamma, gammaFFBS = Step(x0=x0, x1=x1, y0=y0, gradf=gradf, proxhs=proxhs,
                                mulW=mulW, mulWT=mulWT, mulPIn=mulPIn, mulP=mulP, f=f, L=L,
                                dStep=dStep, PReg=PReg, t0=t0, C=C, rho_i=rho(i),
                                kMax=kMax, eps=eps, dInv=dInv)
        elapsed = time.process_time() - start
        if i == 0:
            C = 10 * norm(x1 - x0)
        rre = norm(xOrig - x1) / norm(xOrig)
        rreList.append(rre)
        timeList.append(elapsed)
        ssimList.append(ssim(x1, xOrig, data_range=1))
        gammaList.append(gamma)
        gammaFFBSList.append(gammaFFBS)
        print("Iteration: " + str(i), end="")
        print(", RRE: " + str(rre), end="")
        print(", SSIM: " + str(ssimList[-1]), end="")
        print(", pStep: " + str(eps / L))
        if f(x1) < dp*tol and not dpReached :
            imRec = x1
            dpStopIndex = i+1
            dpReached = True
        if (i+1) in recIndexes:
            recList.append(x1)
            
    if imRec is None:
        imRec = x1
        dpStopIndex = i+1
    return x1, imRec, rreList, ssimList, timeList, gammaList, gammaFFBSList, dpStopIndex, recList

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
