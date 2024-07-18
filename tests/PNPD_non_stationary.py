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

import numpy as np
from math_extras import (
    gradient_convolution_least_squares,
    gradient_2D_signal,
    divergence_2D_signal,
    prox_h_star_TV,
    multiply_P_inverse,
    multiply_P,
    scalar_product,
    convolve_2D_fft,
    total_variation_2D
)
from solvers import PNPD_non_stationary, PNPD_parameters, PNPD_non_stationary_functions, image_metrics
from tests.constants import *
from tests.generate_blurred_image import DeblurProblemData
from dataclasses import dataclass
from utilities import save_data
from plot_extras import TestData, plot_metrics_results
from typing import Callable

TEST_NAME = "PNPD_non_stationary"

@dataclass
class Parameters:
    nu: list[float]
    lam: list[float]
    iterations: int = 10
    k_max: list[int] = None
    bootstrap_iterations = 20

def compute(data: DeblurProblemData, parameters: Parameters, save_path = None):
    methods_parameters = PNPD_parameters(maxIter=parameters.iterations, alpha=1, beta=1/8, kMax=parameters.k_max, extrapolation=True, ground_truth=data.image)

    metrics = image_metrics()

    def nu_scheduler_classic(nu: float):
        return lambda i: .5*.85**i + nu
    
    def nu_scheduler_increasing(nu: float):
        return lambda i: (1-i**(-.5))*(1-nu) + nu
    
    def nu_scheduler_bootstrap(nu_0: float, bootstrap_iterations):
        c = nu_0**-(1/bootstrap_iterations)
        def scheduler(i):
            if i > bootstrap_iterations:
                return 1
            else:
                return c**(i-1-bootstrap_iterations)
        return scheduler
    
    def mul_P_inv_scheduler(nu_scheduler: Callable[[int], float]):
        def scheduler(i: int):
            nu = nu_scheduler(i)
            preconditioner_polynomial = np.polynomial.Polynomial([nu, 1-nu])
            return lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq)
        return scheduler
    
    def mul_P_inv_scheduler_bootstrap(nu_scheduler: Callable[[int], float], bootstrap_iterations: int):
        def scheduler(i: int):
            if i > bootstrap_iterations:
                return lambda x: x
            else:
                nu = nu_scheduler(i)
                preconditioner_polynomial = np.polynomial.Polynomial([nu, 1-nu])
                return lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq)                
        return scheduler
    
    def lam_scheduler_norm_precond(nu_scheduler, lam_bar):
        return lambda i: lam_bar / nu_scheduler(i)
    
    def prox_h_star_scheduler(lam_scheduler: Callable[[int], float]):
        def scheduler(i: int):
            lam = lam_scheduler(i)
            return lambda alpha, x: prox_h_star_TV(lam, x)
        return scheduler
    
    def constant_scheduler(c):
        return lambda i: c
    
    functions = PNPD_non_stationary_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        metrics=metrics
    )

    lam_NPD = 2e-4
    f = lambda x: scalar_product(convolve_2D_fft(x, data.psf) - data.blurred)
    metrics["$\|Ax-b\|_2^2 + \lambda TV(x)$"] = lambda x, ground_truth: f(x) + lam_NPD * total_variation_2D(x)

    im_rec = {}
    metrics_results = {}

    print(TEST_NAME)
    print("\n\n\n\n")

    functions.prox_h_star_scheduler = prox_h_star_scheduler(constant_scheduler(parameters.lam[0]))
    functions.mulP_inv_scheduler = mul_P_inv_scheduler(constant_scheduler(parameters.nu[0]))

    methods_parameters.kMax = parameters.k_max[0]
    method = "PNPD"
    method += f" $\lambda={parameters.lam[0]}$"
    method += f" $k_{{max}}={methods_parameters.kMax}$"
    method += f" $\\nu={parameters.nu[0]}$"
    print(method)
    im_rec_tmp, metrics_results_tmp = PNPD_non_stationary(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")


    functions.prox_h_star_scheduler = prox_h_star_scheduler(lam_scheduler_norm_precond(nu_scheduler_classic(parameters.nu[1]), parameters.lam[1]))
    functions.mulP_inv_scheduler = mul_P_inv_scheduler(nu_scheduler_classic(parameters.nu[1]))
    methods_parameters.reset()
    methods_parameters.kMax = parameters.k_max[1]
    method = "PNPD"
    method += f" $\lambda_k={parameters.lam[1]} /\\nu_k$"
    method += f" $k_{{max}}={methods_parameters.kMax}$"
    method += f" $\\nu_k=0.5 \cdot 0.95^k + {parameters.nu[1]}$"
    print(method)
    im_rec_tmp, metrics_results_tmp = PNPD_non_stationary(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    functions.prox_h_star_scheduler = prox_h_star_scheduler(constant_scheduler(parameters.lam[2]))
    functions.mulP_inv_scheduler = mul_P_inv_scheduler(nu_scheduler_classic(parameters.nu[2]))
    methods_parameters.reset()
    methods_parameters.kMax = parameters.k_max[2]
    method = "PNPD"
    method += f" $\lambda={parameters.lam[2]}$"
    method += f" $k_{{max}}={methods_parameters.kMax}$"
    method += f" $\\nu_k=0.5 \cdot 0.95^k + {parameters.nu[2]}$"
    print(method)
    im_rec_tmp, metrics_results_tmp = PNPD_non_stationary(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    nu_scheduler = nu_scheduler_increasing(parameters.nu[3])
    functions.prox_h_star_scheduler = prox_h_star_scheduler(lam_scheduler_norm_precond(nu_scheduler, parameters.lam[3]))
    functions.mulP_inv_scheduler = mul_P_inv_scheduler(nu_scheduler)
    methods_parameters.reset()
    methods_parameters.kMax = parameters.k_max[3]
    method = "PNPD"
    method += f" $\lambda_k={parameters.lam[3]} /\\nu_k$"
    method += f" $k_{{max}}={methods_parameters.kMax}$"
    method += f" $\\nu_k=(1-\\frac{{1}}{{\\sqrt{{k}}}})(1-\\nu) + {parameters.nu[3]}$"
    print(method)
    im_rec_tmp, metrics_results_tmp = PNPD_non_stationary(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    # TODO this can be further optimized by removing the if statement in the
    # nu bootstrap scheduler
    bootstrap_iterations = parameters.bootstrap_iterations
    nu_scheduler = nu_scheduler_bootstrap(parameters.nu[4], bootstrap_iterations)
    functions.prox_h_star_scheduler = prox_h_star_scheduler(lam_scheduler_norm_precond(nu_scheduler, parameters.lam[4]))
    functions.mulP_inv_scheduler = mul_P_inv_scheduler_bootstrap(nu_scheduler, bootstrap_iterations)
    methods_parameters.reset()
    methods_parameters.kMax = parameters.k_max[4]
    method = "PNPD"
    method += f" $\lambda_k={parameters.lam[4]} /\\nu_k$"
    method += f" $k_{{max}}={methods_parameters.kMax}$"
    method += f" $\\nu_k=$bootstrap"
    print(method)
    im_rec_tmp, metrics_results_tmp = PNPD_non_stationary(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    output_data = TestData(im_rec=im_rec, metrics_results=metrics_results)

    if save_path is not None:
        save_data(save_path, output_data)
    
    return output_data


def plot(data: TestData, save_path = None):
    plot_metrics_results(data.metrics_results, save_path)
    

if __name__ == "__main__":
    from utilities import load_data

    # Load data (generated with generateBlurredImage.py)
    DATA_PATH = "." + PICKLE_SAVE_FOLDER + "/Blurred"
    DATA_SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/" + TEST_NAME
    PLOT_SAVE_PATH = "." + PLOTS_SAVE_FOLDER + "/" + TEST_NAME + "/"
    data = load_data(DATA_PATH)
    parameters = Parameters(nu=1e-1, lam_PNPD=1e-3, lam_NPD=1e-4, iterations=10, k_max=1)
    output_data = compute(data, parameters, DATA_SAVE_PATH)
    plot(output_data, PLOT_SAVE_PATH)
    