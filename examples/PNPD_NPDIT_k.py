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
    multiply_P,
    scalar_product,
    convolve_2D_fft
)
from PNPD.solvers import PNPD, NPDIT_no_backtracking, NPDIT_parameters, NPDIT_functions, image_metrics
from examples.constants import *
from examples.generate_blurred_image import DeblurProblemData
from dataclasses import dataclass
from PNPD.utilities import save_data
from PNPD.plot_extras import TestData, plot_metrics_results

TEST_NAME = "PNPD_NPDIT_k"

@dataclass
class Parameters:
    nu: float
    lam_PNPD: float
    lam_NPD: float
    k_max: list[int]
    iterations: int = 10
    extrapolation: bool = True

def compute(data: DeblurProblemData, parameters: Parameters, save_path = None):
    methods_parameters = NPDIT_parameters(maxIter=parameters.iterations, alpha=1, beta=None, kMax=parameters.k_max, extrapolation=parameters.extrapolation, ground_truth=data.image)

    metrics = image_metrics()

    preconditioner_polynomial = np.polynomial.Polynomial([parameters.nu, 1])

    functions = NPDIT_functions(
        f=lambda x: scalar_product(convolve_2D_fft(x, data.psf) - data.blurred), # TODO convolve_2d_fft should use psfFFT as input
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        prox_h_star=None,
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        mulP_inv=lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq),
        mulP = lambda x: multiply_P(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq),
        metrics=metrics
    )

    im_rec = {}
    metrics_results = {}

    print(TEST_NAME)
    print("\n\n\n\n")

    for k_max in parameters.k_max:
        methods_parameters.reset()

        # Update beta
        methods_parameters.beta = .99/8

        # Update k_{max}
        methods_parameters.kMax = k_max

        # Update \lambda
        lam = parameters.lam_PNPD
        functions.prox_h_star=lambda alpha, x: prox_h_star_TV(lam, x)

        method = "PNPD"
        method += "" if methods_parameters.extrapolation else "_NE"
        method += f", $k_{{max}}={methods_parameters.kMax}$"

        print(method)
        im_rec_tmp, metrics_results_tmp = PNPD(x1=data.blurred, parameters=methods_parameters, functions=functions)
        im_rec[method] = im_rec_tmp
        metrics_results[method] = metrics_results_tmp
        print("\n\n\n\n")

        # Update beta
        methods_parameters.beta *= parameters.nu

        # Update \lambda
        lam = parameters.lam_NPD
        functions.prox_h_star=lambda alpha, x: prox_h_star_TV(lam, x)

        methods_parameters.reset()

        method = "NPDIT"
        method += "" if methods_parameters.extrapolation else "_NE"
        method += f", $k_{{max}}={methods_parameters.kMax}$"

        print(method)
        im_rec_tmp, metrics_results_tmp = NPDIT_no_backtracking(x1=data.blurred, parameters=methods_parameters, functions=functions)
        im_rec[method] = im_rec_tmp
        metrics_results[method] = metrics_results_tmp
        print("\n\n\n\n")

    output_data = TestData(im_rec=im_rec, metrics_results=metrics_results)

    if save_path is not None:
        save_data(save_path, output_data)
    
    return output_data


def plot(data: TestData, save_path = None):
    plot_metrics_results(data.metrics_results, save_path)
