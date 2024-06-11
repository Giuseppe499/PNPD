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

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams['font.family'] = 'STIXGeneral'
props = dict(boxstyle='square', fc="white", ec="black" , alpha=0.5)
boxPos = (0.95, 0.98)

def generate_save_path(filename, directory="./Plots/"):
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

def relative_time_to_absolute(timeList):
    absTimeList = np.zeros(len(timeList))
    absTimeList[0] = timeList[0]
    for i in range(1, len(timeList)):
        absTimeList[i] = absTimeList[i-1] + timeList[i]
    return absTimeList

def is_RGB(image):
     dim = len(image.shape)
     if dim == 2:
          return False
     elif dim == 3:
          return True
     else:
          raise Exception(f"Only grayscale and RGB images supported: input image has {dim} dimensions")
    
def plot_image_psf_blurred(image, psf_centered, blurred):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmap = None if is_RGB(image) else "gray"
    axs[0].imshow(image, cmap=cmap, vmin=0, vmax=1)
    axs[0].set_title("Original")
    axs[1].imshow(psf_centered / psf_centered.max(), cmap=cmap)
    axs[1].set_title("PSF")
    axs[2].imshow(blurred, cmap=cmap, vmin=0, vmax=1)
    axs[2].set_title("Blurred")

def plot_lists(Y, X = None, stopIndices = None, labels = None, labelsStop = None, title = None, xlabel = None, ylabel = None, saveName = None, linestyle = None, semilogy = False):
        plt.figure()
        if X is None:
            X = [np.arange(len(Y[i])) for i in range(len(Y))]
        if labels is None:
            labels = ['' for i in range(len(Y))]
        if linestyle is None:
            linestyle = ['-' for i in range(len(Y))]
        for i in range(len(Y)):
            if semilogy:
                plt.semilogy(X[i], Y[i], label=labels[i], linestyle=linestyle[i])
            else:
                plt.plot(X[i], Y[i], label=labels[i], linestyle=linestyle[i])
        plt.gca().set_prop_cycle(None)
        if stopIndices is not None:
            for i in range(len(stopIndices)):
                plt.plot(X[i][stopIndices[i]], Y[i][stopIndices[i]], 'o', label=labelsStop[i])
        plt.legend()
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        os.makedirs(os.path.dirname(generate_save_path(saveName)), exist_ok=True)
        if saveName is not None:
            plt.savefig(generate_save_path(saveName), bbox_inches='tight')