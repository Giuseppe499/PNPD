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
from mathExtras import (
    gradient_convolution_least_squares,
    gradient_2D_signal,
    divergence_2D_signal,
    prox_h_star_TV,
    multiply_P_inverse,
)
from solvers import PNPD, PNPD_parameters, PNPD_functions, image_metrics

prec_pol_list = [[1e-1, 1],
                 [1e-1, 0, 1],
                 [1e-1, 1, 1]]

# Load data (generated with generateBlurredImage.py)
with np.load(f"./npz/Blurred.npz") as data:
    b = data["b"]
    bFFT = data["bFFT"]
    psf = data["psf"]
    psfFFT = data["psfFFT"]
    psfFFTC = data["psfFFTC"]
    psfAbsSq = data["psfAbsSq"]
    image = data["image"]
    noiseNormSqd = data["noiseNormSqd"]

results = []
for prec_pol in prec_pol_list:
    parameters = PNPD_parameters(maxIter=100, alpha=1, beta=1 / 8, kMax=1, extrapolation=True, ground_truth=image)

    lam = 1e-3  # TV regularization parameter

    metrics = image_metrics()

    preconditioner_polynomial = np.polynomial.Polynomial(prec_pol)

    functions = PNPD_functions(
        grad_f=lambda x: gradient_convolution_least_squares(x, bFFT, psfFFT, psfFFTC),
        prox_h_star=lambda alpha, x: prox_h_star_TV(lam, x),
        mulW=gradient_2D_signal,
        mulWT=divergence_2D_signal,
        mulP_inv= lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=psfAbsSq),
        metrics=image_metrics()
    )

    ################################################################################
    # NPD
    print("PNPD")
    results.append(PNPD(x1=b, parameters=parameters, functions=functions))
    print("\n\n\n\n")

def poly_to_string(p):
    return "$p = " + str(np.polynomial.Polynomial(p)) + "$"

Plot = True
if Plot:
    import matplotlib.pyplot as plt
    
    for key in results[0][1].keys():
        plt.figure()
        for result, pol in zip(results, prec_pol_list):
            value = result[1][key]
            plt.plot(value, label=poly_to_string(pol))
        plt.legend()
        plt.title(key)
    plt.show()
