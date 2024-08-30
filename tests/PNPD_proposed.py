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
    scalar_product,
)
from solvers import (PNPD, PNPD_non_stationary, image_metrics, PNPD_parameters,
                     PNPD_functions, PNPD_non_stationary_functions)
from tests.constants import *
from tests.generate_blurred_image import DeblurProblemData
from dataclasses import dataclass
from utilities import save_data
from plot_extras import TestData, plot_metrics_results, plot_images
import matplotlib.pyplot as plt
from schedulers import *

TEST_NAME = "PNPD_proposed"

@dataclass
class Parameters:
    nu: list[float]
    lam_PNPD: list[float]
    lam_NPD: float
    iterations: int = 10
    k_max: list[int] = None
    bootstrap_iterations: int = 20

def compute(data: DeblurProblemData, parameters: Parameters, save_path = None):
    methods_parameters = PNPD_parameters(maxIter=parameters.iterations, alpha=1, beta=1/8, extrapolation=True, ground_truth=data.image)

    metrics = image_metrics()

    functions = PNPD_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        metrics=metrics
    )

    im_rec = {}
    metrics_results = {}

    print(TEST_NAME)
    print("\n\n\n\n")

    method = ""
    method = "PNPD, "
    method += f"$\\nu={parameters.nu[0]},$"
    method += f" $\lambda={parameters.lam_PNPD[0]}$,"
    method += f" $k_{{max}}={parameters.k_max[0]}$"
    print(method)

    # Set method parameters
    methods_parameters.kMax = parameters.k_max[0]
    preconditioner_polynomial = np.polynomial.Polynomial([parameters.nu[0], 1])
    functions.mulP_inv= lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq)
    functions.prox_h_star=lambda alpha, x: prox_h_star_TV(parameters.lam_PNPD[0], x)

    e = np.random.randn(*data.blurred.shape)
    s = scalar_product(e, functions.mulP_inv(e))/scalar_product(e,e)
    lam_PNPD = parameters.lam_NPD*s
    print(f"lam_PNPD given: {parameters.lam_PNPD[0]}, estimated lam_PNPD: {lam_PNPD}")

    im_rec_tmp, metrics_results_tmp = PNPD(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    method = ""
    method = "PNPD_NE, "
    method += f"$\\nu={parameters.nu[1]},$"
    method += f" $\lambda={parameters.lam_PNPD[1]}$,"
    method += f" $k_{{max}}={parameters.k_max[1]}$"
    print(method)

    # Set method parameters
    methods_parameters.kMax = parameters.k_max[1]
    preconditioner_polynomial = np.polynomial.Polynomial([parameters.nu[1], 1])
    functions.mulP_inv= lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq)
    functions.prox_h_star=lambda alpha, x: prox_h_star_TV(parameters.lam_PNPD[1], x)
    methods_parameters.reset()
    methods_parameters.extrapolation = False

    e = np.random.randn(*data.blurred.shape)
    s = scalar_product(e, functions.mulP_inv(e))/scalar_product(e,e)
    lam_PNPD = parameters.lam_NPD*s
    print(f"lam_PNPD given: {parameters.lam_PNPD[1]}, estimated lam_PNPD: {lam_PNPD}")

    im_rec_tmp, metrics_results_tmp = PNPD(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")
    
    method = ""
    method = "PNPD, "
    method += f"$\\nu={parameters.nu[2]},$"
    method += f" $\lambda={parameters.lam_PNPD[2]}$,"
    method += f" $k_{{max}}={parameters.k_max[2]}$"
    print(method)

    # Set method parameters
    methods_parameters.kMax = parameters.k_max[2]
    preconditioner_polynomial = np.polynomial.Polynomial([parameters.nu[2], 1])
    functions.mulP_inv= lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq)
    functions.prox_h_star=lambda alpha, x: prox_h_star_TV(parameters.lam_PNPD[2], x)
    methods_parameters.reset()
    methods_parameters.extrapolation = True

    e = np.random.randn(*data.blurred.shape)
    s = scalar_product(e, functions.mulP_inv(e))/scalar_product(e,e)
    lam_PNPD = parameters.lam_NPD*s
    print(f"lam_PNPD given: {parameters.lam_PNPD[2]}, estimated lam_PNPD: {lam_PNPD}")

    im_rec_tmp, metrics_results_tmp = PNPD(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    # PNPD Bootstrap
    functions = PNPD_non_stationary_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        metrics=metrics
    )

    method = ""
    method = "PNPD, "
    method += f"$\\nu_n=(5.20)$,"
    method += f"$\\nu_0={parameters.nu[3]}$,"
    method += f" $\lambda_n={parameters.lam_NPD} \cdot \|S^{{-1}}_n\|_2$,"
    method += f" $k_{{max}}={parameters.k_max[3]}$,"
    method += f" $n_{{\\text{{bt}}}}={parameters.bootstrap_iterations}$"
    print(method)

    # Set method parameters
    # TODO this can be further optimized by removing the if statement in the
    # nu bootstrap scheduler
    bootstrap_iterations = parameters.bootstrap_iterations
    nu_scheduler = nu_scheduler_bootstrap(
        parameters.nu[3],
        bootstrap_iterations
        )
    functions.prox_h_star_scheduler = prox_h_star_scheduler(
        lam_scheduler_norm_precond(nu_scheduler, parameters.lam_NPD)
        )
    functions.mulP_inv_scheduler = mul_P_inv_scheduler_bootstrap(
        nu_scheduler,
        data.psfAbsSq,
        bootstrap_iterations
        )
    methods_parameters.reset()
    methods_parameters.kMax = parameters.k_max[3]

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
    plot_reconstructions(data, save_path)


def plot_reconstructions(data: TestData, save_path = None):
    for method, rec in data.im_rec.items():
        iterations = len(data.metrics_results[method]["time"])-1
        plot_images([rec])
        plt.savefig(save_path + f"reconstruction_it={iterations}_{method}.pdf")
    
    

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
    