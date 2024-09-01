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
from schedulers import *

TEST_NAME = "PNPD_bootstrap"

@dataclass
class Parameters:
    nu: list[float]
    lam: float
    iterations: int = 10
    k_max: list[int] = None
    bootstrap_iterations: list[int] = None

def compute(data: DeblurProblemData, parameters: Parameters, save_path = None):
    methods_parameters = PNPD_parameters(maxIter=parameters.iterations, alpha=1, beta=1/8, kMax=parameters.k_max, extrapolation=True, ground_truth=data.image)

    metrics = image_metrics()
    
    functions = PNPD_non_stationary_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        metrics=metrics
    )

    lam_NPD = 2e-4
    f = lambda x: scalar_product(convolve_2D_fft(x, data.psf) - data.blurred)
    metrics["NPD objective function"] = lambda x, ground_truth: f(x) + lam_NPD * total_variation_2D(x)

    im_rec = {}
    metrics_results = {}

    print(TEST_NAME)
    print("\n\n\n\n")

    for i in range(len(parameters.nu)):
        # TODO this can be further optimized by removing the if statement in the
        # nu bootstrap scheduler
        bootstrap_iterations = parameters.bootstrap_iterations[i]
        nu_scheduler = nu_scheduler_bootstrap(
            parameters.nu[i],
            bootstrap_iterations
            )
        functions.prox_h_star_scheduler = prox_h_star_scheduler(
            lam_scheduler_norm_precond(nu_scheduler, parameters.lam)
            )
        functions.mulP_inv_scheduler = mul_P_inv_scheduler_bootstrap(
            nu_scheduler,
            data.psfAbsSq,
            bootstrap_iterations
            )
        methods_parameters.reset()
        methods_parameters.kMax = parameters.k_max[i]
        method = ""
        # method = "PNPD, "
        method += f"$n_\\text{{bt}}={bootstrap_iterations}$,"
        method += f" $\\nu_0={parameters.nu[i]}$,"
        # method += f" $\lambda_k={parameters.lam} /\\nu_k$,"
        # method += f" $k_{{max}}={methods_parameters.kMax}$"
        print(method)
        im_rec_tmp, metrics_results_tmp = PNPD_non_stationary(
            x1=data.blurred,
            parameters=methods_parameters,
            functions=functions
            )
        im_rec[method] = im_rec_tmp
        metrics_results[method] = metrics_results_tmp
        print("\n\n\n\n")

    output_data = TestData(im_rec=im_rec, metrics_results=metrics_results)

    if save_path is not None:
        save_data(save_path, output_data)
    
    return output_data


def plot(data: TestData, save_path = None):
    of_x_hat = float("inf")
    for method in data.metrics_results:
        of = data.metrics_results[method]["NPD objective function"]
        if of[-1] <  of_x_hat:
            of_x_hat = of[-1]
    for method in data.metrics_results:
        of = data.metrics_results[method]["NPD objective function"]
        of = np.abs(of-of_x_hat)/np.abs(of_x_hat)
        data.metrics_results[method]["NPD relative objective function"] = of
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