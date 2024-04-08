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
from mathExtras import scalar_product
from dataclasses import dataclass

import time


def deblur_image_naive(blurred_image, psf, epsilon: float):
    return np.real(ifft2(fft2(blurred_image) / np.clip(fft2(psf), epsilon, None)))


def deblur_image_Thikonov(blurred_image, psf, alpha):
    psf_FFT = fft2(psf)
    psf_FFTC = np.conjugate(psf_FFT)
    blurred_image_FFT = fft2(blurred_image)
    return np.real(ifft2(psf_FFTC * blurred_image_FFT / (psf_FFTC * psf_FFT + alpha)))


def deblur_image_Thikonov_with_prediction(blurred_image, psf, predicted_image, alpha):
    psf_FFT = fft2(psf)
    psf_FFTC = np.conjugate(psf_FFT)
    predicted_image_FFT = fft2(predicted_image)
    blurred_image_FFT = fft2(blurred_image)
    return np.real(
        ifft2(
            (psf_FFTC * blurred_image_FFT + alpha * predicted_image_FFT)
            / (psf_FFTC * psf_FFT + alpha)
        )
    )


@dataclass
class FBS_functions:
    grad_f: Callable[[np.ndarray], np.ndarray]
    prox_g: Callable[[float, np.ndarray], np.ndarray] = None


@dataclass
class FBS_parameters:
    alpha: float
    maxIter: int
    startIteration: int = 1
    iteration: int = None


def FBS(x1: np.ndarray, parameters: FBS_parameters, functions: FBS_functions):
    return genericFBS(x1, parameters, functions, FBS_step)


def genericFBS(
    x1: np.ndarray,
    parameters: FBS_parameters,
    functions: FBS_functions,
    step: Callable[
        [np.ndarray, FBS_parameters, FBS_functions], tuple[np.ndarray, ...]
    ],
):
    for parameters.iteration in range(
        parameters.startIteration, parameters.maxIter
    ):
        x1 = step(x1, parameters, functions)[0]
    return x1


def FBS_step(
    x1: np.ndarray, parameters: FBS_parameters, functions: FBS_functions
):
    return functions.prox_g(
        parameters.alpha, x1 - parameters.alpha * functions.grad_f(x1)
    )


@dataclass
class FFBS_parameters(FBS_parameters):
    x0: np.ndarray = None
    t0: float = 1


def FFBS(x1: np.ndarray, parameters: FFBS_parameters, functions: FBS_functions):
    return genericFBS(x1, parameters, functions, FFBS_step)


def FFBS_step(
    x1: np.ndarray, parameters: FFBS_parameters, functions: FBS_functions
):
    gamma, t1 = gammaFFBS(parameters.t0)
    parameters.t0 = t1
    extrapolatedPoint = computeExtraPoint(x1, parameters.x0, gamma)
    parameters.x0 = x1
    return FBS_step(extrapolatedPoint, parameters, functions), parameters


def gammaFFBS(t0: float):
    t1 = 0.5 * (1 + np.sqrt(1 + 4 * t0 * t0))
    gamma = (t0 - 1) / t1
    return gamma, t1


def computeExtraPoint(x1: np.ndarray, x0: np.ndarray, gamma: float):
    return x1 + gamma * (x1 - x0)


@dataclass
class NPD_parameters(FFBS_parameters):
    beta: float = None
    kMax: int = 1
    C: float = 1
    rho_i: float = 1
    extrapolation: bool = True
    y0 = None


@dataclass
class NPD_functions(FBS_functions):
    prox_h_star: Callable[[float, np.ndarray], np.ndarray] = None
    mulW: Callable[[np.ndarray], np.ndarray] = None
    mulWT: Callable[[np.ndarray], np.ndarray] = None
    rho: Callable[[int], float] = lambda i: 1 / (i + 1) ** 1.1

def NPD_no_extrapolation_step(
    x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions
):
    parameters.rho_i = functions.rho(parameters.iteration)
    return FBS_step(x1, parameters, functions), parameters

def primal_dual_prox_estimator(
    alpha: float,
    x1: np.ndarray,
    parameters: NPD_parameters,
    functions: NPD_functions,
):
    gradf_x1 = functions.grad_f(x1)
    x2, parameters = primal_dual_step(x1, gradf_x1, parameters, functions)
    xSum = np.zeros(x1.shape)
    for k in range(1, parameters.kMax):
        x2, parameters = primal_dual_step(x1, gradf_x1, parameters, functions)
        xSum += x2
    x2, parameters = primal_dual_step(x1, gradf_x1, parameters, functions)
    xSum += x2
    return xSum / parameters.kMax, parameters


def primal_step(
    x1: np.ndarray,
    gradf_x1: np.ndarray,
    parameters: NPD_parameters,
    functions: NPD_functions,
):
    x2 = x1 - parameters.alpha * (gradf_x1 + functions.mulWT(parameters.y0))
    return x2


def primal_dual_step(
    x1: np.ndarray,
    gradf_x1: np.ndarray,
    parameters: NPD_parameters,
    functions: NPD_functions,
):
    x2 = primal_step(x1, gradf_x1, parameters, functions)
    stepsize = parameters.beta / parameters.alpha
    y1 = functions.prox_h_star(
        stepsize, parameters.y0 + stepsize * functions.mulW(x2)
    )
    parameters.y0 = y1
    return x2, parameters


def NPD_step(
    x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions
):
    gamma, t1 = gammaNPD(
        parameters.t0, parameters.C, parameters.rho_i, norm(x1 - parameters.x0)
    )
    parameters.t0 = t1
    extrapolatedPoint = computeExtraPoint(x1, parameters.x0, gamma)
    parameters.x0 = x1
    return NPD_no_extrapolation_step(extrapolatedPoint, parameters, functions)[0], parameters

def gammaNPD(t0: float, C: float, rho_i: float, xDiffNorm: float):
    gamma, t1 = gammaFFBS(t0)
    gamma = min(gamma, C * rho_i / xDiffNorm)
    return gamma, t1

def NPD(x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions):
    parameters.x0 = x1
    parameters.y0 = np.zeros(functions.mulW(x1).shape)
    functions.prox_g = lambda alpha, x: primal_dual_prox_estimator(
        alpha, x, parameters, functions
    )[0]
    if parameters.extrapolation:
        step = NPD_step
    else:
        step = NPD_no_extrapolation_step

    # First step: needed to compute C
    parameters.iteration = 1
    x1 = step(x1, parameters, functions)[0]
    parameters.C = 10 * norm(x1 - parameters.x0)
    parameters.startIteration = 2
    return genericFBS(x1, parameters, functions, step)

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
        if f(x2) <= f(xBar) + scalar_product(gradf_xBar, x2MinusXBar) + L / 2 * scalar_product(x2MinusXBar, mulP(PReg, x2MinusXBar)):
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
        if f(x2) <= f(x1) + scalar_product(gradf_x1, x2MinusX1) + L / 2 * scalar_product(x2MinusX1, mulP(PReg, x2MinusX1)):
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