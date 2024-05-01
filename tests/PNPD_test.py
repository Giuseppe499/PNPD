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

parameters = PNPD_parameters(maxIter=100, alpha=1, beta=1 / 8, kMax=1, extrapolation=True, ground_truth=image)

lam = 1e-3  # TV regularization parameter

metrics = image_metrics()

preconditioner_polynomial = np.polynomial.Polynomial([1e-1, 1])

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
imRec, metrics_results = PNPD(x1=b, parameters=parameters, functions=functions)
print("\n\n\n\n")

np.savez(f"./npz/NPD.npz", imRec=imRec)


Plot = True
if Plot:
    import matplotlib.pyplot as plt

    with np.load(f"./npz/Blurred.npz") as data:
        image = data["image"]
        if image.shape.__len__() == 3 and image.shape[2] == 3:
            RGB = True
        else:
            RGB = False

    cmap = "gray" if not RGB else None
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title("Original image")
    ax[0].axis("off")
    ax[1].imshow(imRec, cmap=cmap, vmin=0, vmax=1)
    ax[1].set_title("Recovered image")
    ax[1].axis("off")

    
    for key, value in metrics_results.items():
        plt.figure()
        plt.plot(value, label=key)
        plt.legend()
        plt.title(key + " " + str(preconditioner_polynomial))
    plt.show()
