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
    """Functions for the Forward-Backward Splitting algorithm."""
    grad_f: Callable[[np.ndarray], np.ndarray]
    prox_g: Callable[[float, np.ndarray], np.ndarray] = None
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = None


@dataclass
class FBS_parameters:
    """Parameters for the Forward-Backward Splitting algorithm."""
    alpha: float
    maxIter: int
    verbose: bool = True
    startIteration: int = 1
    iteration: int = None
    ground_truth: np.ndarray = None

    def reset(self):
        self.iteration = None

def FBS(x1: np.ndarray, parameters: FBS_parameters, functions: FBS_functions):
    """Forward-Backward Splitting algorithm."""
    return genericFBS(x1, parameters, functions, FBS_step)

def genericFBS(
    x1: np.ndarray,
    parameters: FBS_parameters,
    functions: FBS_functions,
    step: Callable[
        [np.ndarray, FBS_parameters, FBS_functions], tuple[np.ndarray, ...]
    ],
):
    """Generic implementation of a Forward-Backward Splitting algorithm."""
    metrics_results = None
    metrics_flag = functions.metrics is not None
    if metrics_flag:
        old_step = step
        step = lambda x1, par, fun: metrics_decorator(old_step)(x1, par, fun, metrics_functions=functions.metrics, ground_truth=parameters.ground_truth)
        metrics_results = initialize_metrics_dict(x1, functions.metrics, parameters.ground_truth)
    for parameters.iteration in range(
        parameters.startIteration, parameters.maxIter
    ):
        tmp = step(x1, parameters, functions)
        info = f"Iteration {parameters.iteration}, "
        if metrics_flag:
            x1 = tmp[0][0]
            new_metrics_results = tmp[1]
            info += info_string_from_metrics(new_metrics_results)
            print(info) if parameters.verbose else None
            update_metrics_dict(metrics_results, new_metrics_results)           
        else:
            x1 = tmp[0]

    return x1, metrics_results

def metrics_decorator(step):
    """Decorator to add metrics computation to the step of an iterative algorithm."""
    def wrapper(*args, **kwargs):
        metrics_functions = kwargs.get("metrics_functions", None)
        kwargs.pop("metrics_functions", None)
        ground_truth = kwargs.get("ground_truth", None)
        kwargs.pop("ground_truth", None)

        # Time the function
        start = time.process_time()
        result = step(*args, **kwargs)
        elapsed = time.process_time() - start

        # Compute metrics
        metrics_results = {}
        metrics_results["time"] = elapsed
        for key, value in metrics_functions.items():
            metrics_results[key] = value(result[0], ground_truth=ground_truth)

        return result, metrics_results
    return wrapper

def image_metrics():
    """Return a dictionary of image metrics functions."""
    return {
        "SSIM": lambda x, ground_truth: ssim(x, ground_truth, data_range=1),
        "RRE": lambda x, ground_truth: norm(x - ground_truth) / norm(ground_truth),
    }

def initialize_metrics_dict(x1, metrics_functions, ground_truth):
    """Initialize a dictionary of metrics results."""
    metrics_results = {}
    metrics_results["time"] = [0]
    for key, value in metrics_functions.items():
        metrics_results[key] = [value(x1, ground_truth=ground_truth)]
    return metrics_results

def update_metrics_dict(metrics_results, new_metrics_results):
    """Update a dictionary of metrics results."""
    for key, value in new_metrics_results.items():
            metrics_results[key].append(value)

def info_string_from_metrics(new_metrics_results):
    """Return a string with the information from the metrics results."""
    info = ""
    for key, value in new_metrics_results.items():
        info += f"{key}: {value}, "
    return info

def gradient_descent_step(
    x1: np.ndarray, parameters: FBS_parameters, functions: FBS_functions
):
    """Gradient descent step."""
    return x1 - parameters.alpha * functions.grad_f(x1)

def FBS_step(
    x1: np.ndarray, parameters: FBS_parameters, functions: FBS_functions, descent_step: Callable = gradient_descent_step
):
    """Forward-Backward Splitting step."""
    return functions.prox_g(
        parameters.alpha, descent_step(x1, parameters, functions)
    )

@dataclass
class FFBS_parameters(FBS_parameters):
    """Parameters for the Fast Forward-Backward Splitting algorithm."""
    x0: np.ndarray = None
    t0: float = 1

    def reset(self):
        super().reset()
        self.x0 = None
        self.t0 = 1


def FFBS(x1: np.ndarray, parameters: FFBS_parameters, functions: FBS_functions):
    """Fast Forward-Backward Splitting algorithm."""
    return genericFBS(x1, parameters, functions, FFBS_step)


def FFBS_step(
    x1: np.ndarray, parameters: FFBS_parameters, functions: FBS_functions
):
    """Fast Forward-Backward Splitting step."""
    gamma, t1 = gammaFFBS(parameters.t0)
    parameters.t0 = t1
    extrapolatedPoint = computeExtraPoint(x1, parameters.x0, gamma)
    parameters.x0 = x1
    return FBS_step(extrapolatedPoint, parameters, functions), parameters


def gammaFFBS(t0: float):
    """Compute the gamma and t1 for the Fast Forward-Backward Splitting algorithm."""
    t1 = 0.5 * (1 + np.sqrt(1 + 4 * t0 * t0))
    gamma = (t0 - 1) / t1
    return gamma, t1


def computeExtraPoint(x1: np.ndarray, x0: np.ndarray, gamma: float):
    """Compute an extrapolated point on the line between x0 and x1."""
    return x1 + gamma * (x1 - x0)


@dataclass
class NPD_parameters(FFBS_parameters):
    """Parameters for the Nested Primal-Dual algorithm."""
    beta: float = None
    kMax: int = 1
    C: float = 1
    rho_i: float = 1
    extrapolation: bool = True
    y1 = None
    grad_f_x1 = None

    def reset(self):
        super().reset()
        self.C = 1
        self.rho_i = 1
        self.y1 = None
        self.grad_f_x1 = None


@dataclass
class NPD_functions(FBS_functions):
    """Functions for the Nested Primal-Dual algorithm."""
    prox_h_star: Callable[[float, np.ndarray], np.ndarray] = None
    mulW: Callable[[np.ndarray], np.ndarray] = None
    mulWT: Callable[[np.ndarray], np.ndarray] = None
    rho: Callable[[int], float] = lambda i: 1 / (i + 1) ** 1.1

def NPD_no_extrapolation_step(
    x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions, descent_step: Callable = gradient_descent_step):
    """Nested Primal-Dual step without extrapolation."""
    parameters.rho_i = functions.rho(parameters.iteration)
    return FBS_step(x1, parameters, functions, descent_step), parameters

class NPD_prox_estimator:
    """Class for the Nested Primal-Dual proximal estimator."""

    @classmethod
    def primal_dual_prox_estimator(
        cls,
        alpha: float,
        x: np.ndarray,
        parameters: NPD_parameters,
        functions: NPD_functions,
    ):
        """Nested Primal-Dual proximal estimator."""
        x2, parameters = cls.primal_dual_step(x, parameters, functions)
        xSum = np.zeros(x.shape)
        for k in range(1, parameters.kMax):
            x2, parameters = cls.primal_dual_step(x, parameters, functions)
            xSum += x2
        x2 = cls.primal_step(x, parameters, functions)
        xSum += x2
        return xSum / parameters.kMax, parameters
    
    @classmethod
    def primal_step(
        cls,
        x: np.ndarray,
        parameters: NPD_parameters,
        functions: NPD_functions
    ):
        x2 = x - parameters.alpha * functions.mulWT(parameters.y1)
        return x2
    
    @classmethod
    def dual_step(
        cls,
        x2: np.ndarray,
        parameters: NPD_parameters,
        functions: NPD_functions
    ):
        stepsize = parameters.beta / parameters.alpha
        y2 = functions.prox_h_star(stepsize, parameters.y1 + stepsize * functions.mulW(x2))
        return y2
    
    @classmethod
    def primal_dual_step(
        cls,
        x: np.ndarray,
        parameters: NPD_parameters,
        functions: NPD_functions,
    ):
        x2 = cls.primal_step(x, parameters, functions)
        y2 = cls.dual_step(x2, parameters, functions)
        parameters.y1 = y2
        return x2, parameters


def NPD_step(
    x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions, descent_step: Callable = gradient_descent_step
):
    """Nested Primal-Dual step."""
    gamma, t1 = gammaNPD(
        parameters.t0, parameters.C, parameters.rho_i, norm(x1 - parameters.x0)
    )
    parameters.t0 = t1
    extrapolatedPoint = computeExtraPoint(x1, parameters.x0, gamma)
    parameters.x0 = x1
    return NPD_no_extrapolation_step(extrapolatedPoint, parameters, functions, descent_step = gradient_descent_step)[0], parameters

def gammaNPD(t0: float, C: float, rho_i: float, xDiffNorm: float):
    """Compute the gamma and t1 for the Nested Primal-Dual algorithm."""
    gamma, t1 = gammaFFBS(t0)
    gamma = min(gamma, C * rho_i / xDiffNorm)
    return gamma, t1

def NPD(x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions):
    """Nested Primal-Dual algorithm."""
    return generic_NPD(x1, parameters, functions, NPD_prox_estimator.primal_dual_prox_estimator
    )

def generic_NPD(x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions, prox_estimator: Callable, descent_step: Callable = gradient_descent_step):
    """Generic implementation of the Nested Primal-Dual algorithm."""
    parameters.x0 = x1
    parameters.y1 = np.zeros(functions.mulW(x1).shape)
    functions.prox_g = lambda alpha, x: prox_estimator(
        alpha, x, parameters, functions
    )[0]
    if parameters.extrapolation:
        step = lambda x1, par, fun: NPD_step(x1, par, fun, descent_step=descent_step)
    else:
        step = lambda x1, par, fun: NPD_no_extrapolation_step(x1, par, fun, descent_step=descent_step)

    # First step: needed to compute C
    parameters.iteration = parameters.startIteration
    x1 = step(x1, parameters, functions)[0]
    parameters.C = 10 * norm(x1 - parameters.x0)
    parameters.startIteration += 1
    result = genericFBS(x1, parameters, functions, step)
    parameters.startIteration -= 1
    return result

@dataclass
class PNPD_parameters(NPD_parameters):
    """Parameters for the Preconditioned Nested Primal-Dual algorithm."""
    mulP_x = None

@dataclass
class PNPD_functions(NPD_functions):
    """Functions for the Preconditioned Nested Primal-Dual algorithm."""
    mulP_inv: Callable[[np.ndarray], np.ndarray] = None
    
def preconditioned_gradient_descent_step(
    x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions
):
    """Preconditioned gradient descent step."""
    return x1 - parameters.alpha * functions.mulP_inv(functions.grad_f(x1))

def PNPD(x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions):
    """Preconditioned Nested Primal-Dual algorithm."""
    return generic_NPD(x1, parameters, functions, NPD_prox_estimator.primal_dual_prox_estimator, preconditioned_gradient_descent_step)

class NPDIT_prox_estimator(NPD_prox_estimator):
    """Class for the Nested Primal-Dual Iterated Tikhonov proximal estimator."""

    @classmethod
    def primal_step(
        cls,
        x: np.ndarray,
        parameters: PNPD_parameters,
        functions: PNPD_functions,
    ):
        x2 = x - parameters.alpha * functions.mulP_inv(functions.mulWT(parameters.y1))
        return x2

def NPDIT_no_backtracking(x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions):
    """Nested Primal-Dual Iterated Tikhonov step without backtracking."""
    return generic_NPD(x1, parameters, functions, NPDIT_prox_estimator.primal_dual_prox_estimator, preconditioned_gradient_descent_step)