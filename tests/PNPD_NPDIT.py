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
from solvers import PNPD, NPD, NPDIT_no_backtracking, PNPD_parameters, PNPD_functions, image_metrics

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

parameters = PNPD_parameters(maxIter=150, alpha=.99, beta=.99 / 8, kMax=1, extrapolation=True, ground_truth=image)

lam = 1e-3  # TV regularization parameter

metrics = image_metrics()

nu = 1e-1
preconditioner_polynomial = np.polynomial.Polynomial([nu, 1])

functions = PNPD_functions(
    grad_f=lambda x: gradient_convolution_least_squares(x, bFFT, psfFFT, psfFFTC),
    prox_h_star=lambda alpha, x: prox_h_star_TV(lam, x),
    mulW=gradient_2D_signal,
    mulWT=divergence_2D_signal,
    mulP_inv= lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=psfAbsSq),
    metrics=metrics
)

imRec = {}
metrics_results = {}

method = "PNPD"
print(method)
im_rec_tmp, metrics_results_tmp = PNPD(x1=b, parameters=parameters, functions=functions)
imRec[method] = im_rec_tmp
metrics_results[method] = metrics_results_tmp
print("\n\n\n\n")

method = "NPD"
parameters.reset()
lam = 1e-4
functions.prox_h_star = lambda alpha, x: prox_h_star_TV(lam, x)

print(method)
im_rec_tmp, metrics_results_tmp = NPD(x1=b, parameters=parameters, functions=functions)
imRec[method] = im_rec_tmp
metrics_results[method] = metrics_results_tmp
print("\n\n\n\n")

method = "NPDIT"
parameters.reset()
parameters.beta *= nu

print(method)
im_rec_tmp, metrics_results_tmp = NPDIT_no_backtracking(x1=b, parameters=parameters, functions=functions)
imRec[method] = im_rec_tmp
metrics_results[method] = metrics_results_tmp
print("\n\n\n\n")


PLOT = True
PLOT_RECONSTRUCTION = False
if PLOT:
    import matplotlib.pyplot as plt

    with np.load(f"./npz/Blurred.npz") as data:
        image = data["image"]
        if image.shape.__len__() == 3 and image.shape[2] == 3:
            RGB = True
        else:
            RGB = False

    cmap = "gray" if not RGB else None

    # Results vs iterations
    for key in metrics.keys():
        plt.figure()
        for method in imRec.keys():
            plt.plot(metrics_results[method][key], label=method)
        plt.legend()
        plt.title(key + " vs iterations")

    # Change time from relative to absolute
    from plotExtras import relTimetoAbsTime
    for method in imRec.keys():
        metrics_results[method]["time"] = relTimetoAbsTime(metrics_results[method]["time"])

    # Results vs time
    for key in metrics.keys():
        plt.figure()
        for method in imRec.keys():
            plt.plot(metrics_results[method]["time"], metrics_results[method][key], label=method)
        plt.legend()
        plt.title(key + " vs time")

    
    
    if PLOT_RECONSTRUCTION:
        for method in imRec.keys():
            plt.figure()
            plt.imshow(imRec[method], cmap=cmap)
            plt.title(method)

    plt.show()