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

from context import PNPD

import numpy as np
from PNPD.math_extras import (
    gradient_convolution_least_squares,
    gradient_2D_signal,
    divergence_2D_signal,
    prox_h_star_TV,
    multiply_P_inverse,
    scalar_product,
    total_variation_2D,
    convolve_2D_fft
)
from PNPD.solvers import PNPD, PNPD_parameters, PNPD_functions, image_metrics
from examples.constants import *
from examples.generate_blurred_image import DeblurProblemData
from dataclasses import dataclass
from PNPD.utilities import save_data
from PNPD.plot_extras import TestData, plot_metrics_results

TEST_NAME = "PNPD_nu"

@dataclass
class Parameters:
    nu: list[float]
    lam_PNPD: list[float]
    k_max: list[int]
    iterations: int = 10
    extrapolation: list[bool] = True
    k_max_in_method_name: bool = True

def compute(data: DeblurProblemData, parameters: Parameters, save_path = None):
    methods_parameters = PNPD_parameters(maxIter=parameters.iterations, alpha=1, beta=.99/8, kMax=None, extrapolation=parameters.extrapolation, ground_truth=data.image)

    metrics = image_metrics()

    functions = PNPD_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        prox_h_star=None,
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        mulP_inv=None,
        metrics=metrics
    )

    im_rec = {}
    metrics_results = {}

    print(TEST_NAME)
    print("\n\n\n\n")

    for i, nu in enumerate(parameters.nu):
        methods_parameters.reset()

        # Update k_{max}
        methods_parameters.kMax = parameters.k_max[i]

        # Update \nu
        preconditioner_polynomial = np.polynomial.Polynomial([nu, 1])
        functions.mulP_inv = lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq)

        # Update \lambda
        lam = parameters.lam_PNPD[i]
        functions.prox_h_star=lambda alpha, x: prox_h_star_TV(lam, x)

        # Update extrapolation
        methods_parameters.extrapolation = parameters.extrapolation[i]

        # Update metrics
        res = lambda x: convolve_2D_fft(x, data.psf) - data.blurred
        f_S = lambda x: scalar_product(res(x),functions.mulP_inv(res(x)))
        metrics["PNPD objective function"] = lambda x, ground_truth: f_S(x) + lam * total_variation_2D(x)

        method = ""
        # method = "PNPD"
        # method += "" if methods_parameters.extrapolation else "_NE"
        method += f"$\\nu={nu}$"
        method += f", $\lambda={lam}$"
        if parameters.k_max_in_method_name:
            method += f", $k_{{max}}={methods_parameters.kMax}$"

        print(method)
        im_rec_tmp, metrics_results_tmp = PNPD(x1=data.blurred, parameters=methods_parameters, functions=functions)
        im_rec[method] = im_rec_tmp
        metrics_results[method] = metrics_results_tmp
        print("\n\n\n\n")

    output_data = TestData(im_rec=im_rec, metrics_results=metrics_results)

    if save_path is not None:
        save_data(save_path, output_data)
    
    return output_data


def plot(data: TestData, save_path = None):
    for method in data.metrics_results:
        of = data.metrics_results[method]["PNPD objective function"]
        of_x_hat = of[-1]
        of = np.abs(of-of_x_hat)/np.abs(of_x_hat)
        data.metrics_results[method]["PNPD relative objective function"] = of
    plot_metrics_results(data.metrics_results, save_path)
