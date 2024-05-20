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
    multiply_P
)
from solvers import PNPD, PNPD_parameters, PNPD_functions, image_metrics
from tests.constants import *
from tests.generate_blurred_image import DeblurProblemData
from dataclasses import dataclass
from utilities import save_data
import matplotlib.pyplot as plt
import os

TEST_NAME = "PNPD_extrapolation"

@dataclass
class Parameters:
    nu: float
    lam_PNPD: float
    iterations: int = 10
    k_max: int = 1

@dataclass
class TestData:
    im_rec: dict
    metrics_results: dict

def compute(data: DeblurProblemData, parameters: Parameters, save_path = None):
    methods_parameters = PNPD_parameters(maxIter=parameters.iterations, alpha=1, beta=1/8, kMax=parameters.k_max, extrapolation=True, ground_truth=data.image)

    metrics = image_metrics()

    preconditioner_polynomial = np.polynomial.Polynomial([parameters.nu, 1])

    functions = PNPD_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, data.bFFT, data.psfFFT, data.psfFFTC),
        prox_h_star=lambda alpha, x: prox_h_star_TV(parameters.lam_PNPD, x),
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        mulP_inv= lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=data.psfAbsSq),
        metrics=metrics
    )

    im_rec = {}
    metrics_results = {}

    print(TEST_NAME)
    print("\n\n\n\n")

    method = "PNPD"
    print(method)
    im_rec_tmp, metrics_results_tmp = PNPD(x1=data.blurred, parameters=methods_parameters, functions=functions)
    im_rec[method] = im_rec_tmp
    metrics_results[method] = metrics_results_tmp
    print("\n\n\n\n")

    method = "PNPD_NE"
    methods_parameters.reset()
    methods_parameters.extrapolation = False
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
    imRec = data.im_rec
    metrics_results = data.metrics_results
    metrics = list(next(iter(metrics_results.values())).keys())
    metrics.remove("time")

    # Results vs iterations
    for key in metrics:
        plt.figure()
        for method in imRec.keys():
            plt.plot(metrics_results[method][key], label=method)
        plt.legend()
        plt.title(key + " vs iterations")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path + key + "_iterations" + ".pdf", bbox_inches='tight')

    # Change time from relative to absolute
    from tests.plot_extras import relative_time_to_absolute
    for method in imRec.keys():
        metrics_results[method]["time"] = relative_time_to_absolute(metrics_results[method]["time"])

    # Results vs time
    for key in metrics:
        plt.figure()
        for method in imRec.keys():
            plt.plot(metrics_results[method]["time"], metrics_results[method][key], label=method)
        plt.legend()
        plt.title(key + " vs time")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path + key + "_time" + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    from utilities import load_data

    # Load data (generated with generateBlurredImage.py)
    DATA_PATH = "." + PICKLE_SAVE_FOLDER + "/Blurred"
    DATA_SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/" + TEST_NAME
    PLOT_SAVE_PATH = "." + PLOTS_SAVE_FOLDER + "/" + TEST_NAME + "/"
    data = load_data(DATA_PATH)
    parameters = Parameters(nu=1e-1, lam_PNPD=1e-3, iterations=10, k_max=1)
    output_data = compute(data, parameters, DATA_SAVE_PATH)
    plot(output_data, PLOT_SAVE_PATH)
    