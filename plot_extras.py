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
from dataclasses import dataclass
import typing
import pandas as pd

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams['font.family'] = 'STIXGeneral'
props = dict(boxstyle='square', fc="white", ec="black" , alpha=0.5)
boxPos = (0.95, 0.98)

metrics_results_type = typing.Dict[str, typing.Dict[str, np.ndarray]]

@dataclass
class TestData:
    im_rec: typing.Dict[str, np.ndarray]
    metrics_results: metrics_results_type

def plot_metrics_results(metrics_results: metrics_results_type, save_folder_path: str = None):
    metrics_results = pd.DataFrame(metrics_results)
    metrics = list(metrics_results.index)
    metrics.remove("time")

    # Results vs iterations
    for key in metrics:
        plot_dict(metrics_results.loc[key].to_dict(), save_path=save_folder_path + key + "_iterations" + ".pdf", xlabel="Iterations", ylabel=key)
        plot_dict(metrics_results.loc[key].to_dict(), save_path=save_folder_path + key + "_semilogy_iterations" + ".pdf", xlabel="Iterations", ylabel=key, plot_function=plt.semilogy)

    # Change time from relative to absolute
    time = metrics_results.loc["time"].to_dict()
    for method in time:
        time[method] = relative_time_to_absolute(time[method])

    # Results vs time
    for key in metrics:
        plot_dict(x=time, y=metrics_results.loc[key].to_dict(), save_path=save_folder_path + key + "_time" + ".pdf", xlabel="Time (s)", ylabel=key)
        plot_dict(x=time, y=metrics_results.loc[key].to_dict(), save_path=save_folder_path + key + "_semilogy_time" + ".pdf", xlabel="Time (s)", ylabel=key, plot_function=plt.semilogy)

def plot_dict(y: typing.Dict[str, np.ndarray], x:typing.Dict[str, np.ndarray]=None, save_path: str = None, title:str = None, xlabel:str = None, ylabel:str = None, plot_function = plt.plot):
    plt.figure()
    for key in y:
        if x is not None:
            plot_function(x[key], y[key], label=key)
        else:
            plot_function(y[key], label=key)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')


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
     
def plot_images(images: list, titles: list = None, shape = None, cmap="gray"):
    if shape is None:
        shape = [1, len(images)]
    if titles is None:
        titles = [None] * len(images)
    figsize=np.array((shape[1],shape[0]))*3.5
    fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize)
    axs = np.matrix(axs)
    for i in range(shape[0]):
        for j in range(shape[1]):
            axs[i,j].imshow(images[i*shape[0]+j], cmap=cmap, vmin=0, vmax=1)
            axs[i,j].set_title(titles[i*shape[0]+j])
    plt.tight_layout()
    
def plot_image_psf_blurred(image, psf_centered, blurred):
    plot_images([image, psf_centered, blurred], ["Original", "PSF", "Blurred"])