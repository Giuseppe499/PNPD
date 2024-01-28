"""
NPD implementation

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
from matplotlib import pyplot as plt

if __name__ == '__main__':
    with np.load('grayscaleBlurred.npz') as data:
        b = data['b']
        psf = data['psf']
        image = data['image']

    with np.load('grayscalePNPD.npz') as data:
        imRecPNPD = data['imRec']
        rreListPNPD = data['rreList']

    with np.load('grayscaleNPD.npz') as data:
        imRecNPD = data['imRec']
        rreListNPD = data['rreList']

    # Plot original image
    plt.figure()
    plt.imshow(image, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Original Image')

    # Plot blurred image
    plt.figure()
    plt.imshow(b, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Blurred Image')

    # Plot PNPD reconstruction
    plt.figure()
    plt.imshow(imRecPNPD, cmap='gray', vmin = 0, vmax = 1)
    plt.title('PNPD Reconstruction')

    # Plot NPD reconstruction
    plt.figure()
    plt.imshow(imRecNPD, cmap='gray', vmin=0, vmax=1)
    plt.title('NPD Reconstruction')

    # Plot relative residual error
    plt.figure()
    plt.semilogy(rreListPNPD, label='PNPD')
    plt.semilogy(rreListNPD, label='NPD')
    plt.legend()
    plt.title('Relative Residual Error')
    plt.show()

