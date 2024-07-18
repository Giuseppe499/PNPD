"""
PNPD implementation

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
from math_extras import scalar_product
from dataclasses import dataclass

import time

def deblur_image_pseudoinverse(blurred_image, psf):
    psf_FFT = fft2(psf)
    pseudoinverse = np.divide(1, psf_FFT,
                              out=np.zeros(psf.shape, dtype=psf_FFT.dtype), where=psf_FFT>1e-15)
    return np.real(ifft2(fft2(blurred_image) * pseudoinverse))

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
        start = time.perf_counter()
        result = step(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Compute metrics
        metrics_results = {}
        metrics_results["time"] = elapsed
        for key, value in metrics_functions.items():
            metrics_results[key] = value(result[0], ground_truth=ground_truth)

        return result, metrics_results
    return wrapper

def image_metrics(channel_axis = None):
    """Return a dictionary of image metrics functions."""
    return {
        "SSIM": lambda x, ground_truth: ssim(x, ground_truth, data_range=1, channel_axis=channel_axis),
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
    gamma, t1 = gamma_FFBS(parameters.t0)
    parameters.t0 = t1
    extrapolatedPoint = compute_extra_point(x1, parameters.x0, gamma)
    parameters.x0 = x1
    return FBS_step(extrapolatedPoint, parameters, functions), parameters


def gamma_FFBS(t0: float):
    """Compute the gamma and t1 for the Fast Forward-Backward Splitting algorithm."""
    t1 = 0.5 * (1 + np.sqrt(1 + 4 * t0 * t0))
    gamma = (t0 - 1) / t1
    return gamma, t1


def compute_extra_point(x1: np.ndarray, x0: np.ndarray, gamma: float):
    """Compute an extrapolated point on the line between x0 and x1."""
    return x1 + gamma * (x1 - x0)


@dataclass
class NPD_parameters(FFBS_parameters):
    """Parameters for the Nested Primal-Dual algorithm."""
    beta: float = None
    kMax: int = 1
    C: float = 1
    extrapolation: bool = True
    y1 = None
    grad_f_x1 = None

    def reset(self):
        super().reset()
        self.C = 1
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
    return FBS_step(x1, parameters, functions, descent_step), parameters

def NPD_extrapolation_decorator(step: Callable):
    def step_with_extrapolation(x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions):
        rho_i = functions.rho(parameters.iteration)
        gamma, t1 = gamma_NPD(
            parameters.t0, parameters.C, rho_i, norm(x1 - parameters.x0)
        )
        parameters.t0 = t1
        extrapolatedPoint = compute_extra_point(x1, parameters.x0, gamma)
        parameters.x0 = x1
        return step(extrapolatedPoint, parameters, functions)
    return step_with_extrapolation

NPD_step = NPD_extrapolation_decorator(NPD_no_extrapolation_step)

def gamma_NPD(t0: float, C: float, rho_i: float, xDiffNorm: float):
    """Compute the gamma and t1 for the Nested Primal-Dual algorithm."""
    gamma, t1 = gamma_FFBS(t0)
    gamma = min(gamma, C * rho_i / xDiffNorm)
    return gamma, t1

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

def NPD(x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions):
    """Nested Primal-Dual algorithm."""
    return generic_NPD(x1, parameters, functions, NPD_prox_estimator.primal_dual_prox_estimator, NPD_no_extrapolation_step)

def generic_NPD(x1: np.ndarray, parameters: NPD_parameters, functions: NPD_functions, prox_estimator: Callable, step: Callable):
    """Generic implementation of the Nested Primal-Dual algorithm."""
    parameters.x0 = x1
    parameters.y1 = np.zeros(functions.mulW(x1).shape)
    functions.prox_g = lambda alpha, x: prox_estimator(
        alpha, x, parameters, functions
    )[0]
    if parameters.extrapolation:
        no_extrapolation_step = step
        step = NPD_extrapolation_decorator(no_extrapolation_step)

    metrics_flag = functions.metrics is not None
    if metrics_flag:
        zero_step_metrics_results = initialize_metrics_dict(x1, functions.metrics, parameters.ground_truth)
        

    # First step: needed to compute C
    parameters.iteration = parameters.startIteration
    start = time.perf_counter()
    x1 = step(x1, parameters, functions)[0]
    elapsed = time.perf_counter() - start

    parameters.C = 10 * norm(x1 - parameters.x0)
    parameters.startIteration += 1
    x1, metrics_results = genericFBS(x1, parameters, functions, step)
    parameters.startIteration -= 1

    if metrics_flag:
        for key in metrics_results.keys():
            metrics_results[key].insert(0, zero_step_metrics_results[key][0])
        metrics_results["time"][1]=elapsed

    
    return x1, metrics_results

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

def PNPD_no_extrapolation_step(x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions):
    return NPD_no_extrapolation_step(x1=x1, parameters=parameters, functions=functions, descent_step=preconditioned_gradient_descent_step)

def PNPD(x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions):
    """Preconditioned Nested Primal-Dual algorithm."""
    return generic_NPD(x1, parameters, functions, NPD_prox_estimator.primal_dual_prox_estimator, PNPD_no_extrapolation_step)

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
    
NPDIT_no_extrapolation_step = PNPD_no_extrapolation_step

def NPDIT_no_backtracking(x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions):
    """Nested Primal-Dual Iterated Tikhonov step without backtracking."""
    return generic_NPD(x1, parameters, functions, NPDIT_prox_estimator.primal_dual_prox_estimator, NPDIT_no_extrapolation_step)

@dataclass
class NPDIT_parameters(PNPD_parameters):
    eps = .99
    delta = .5
    L = .1

@dataclass
class NPDIT_functions(PNPD_functions):
    f: Callable[[np.ndarray], np.ndarray] = None
    mulP: Callable[[np.ndarray], np.ndarray] = None

def NPDIT_step(x1: np.ndarray, parameters: NPDIT_parameters, functions: NPDIT_functions):
    grad_f_x1 = functions.grad_f(x1)
    while True:
        parameters.alpha = parameters.eps / parameters.L
        x2, parameters = NPDIT_no_extrapolation_step(x1, parameters, functions)        
        x2_minus_x1 = x2 - x1
        f = functions.f
        if f(x2) <= f(x1) + scalar_product(grad_f_x1, x2_minus_x1) + parameters.L/2 * scalar_product(x2_minus_x1, functions.mulP(x2_minus_x1)):
            break
        parameters.L /= parameters.delta
    return x2, parameters

def NPDIT(x1: np.ndarray, parameters: PNPD_parameters, functions: PNPD_functions):
    """Nested Primal-Dual Iterated Tikhonov step without backtracking."""
    return generic_NPD(x1, parameters, functions, NPDIT_prox_estimator.primal_dual_prox_estimator, NPDIT_step)