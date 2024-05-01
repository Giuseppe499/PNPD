import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams['font.family'] = 'STIXGeneral'
props = dict(boxstyle='square', fc="white", ec="black" , alpha=0.5)
boxPos = (0.95, 0.98)

def savePath(filename, directory="./Plots/"):
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

def relTimetoAbsTime(timeList):
    absTimeList = np.zeros(len(timeList))
    absTimeList[0] = timeList[0]
    for i in range(1, len(timeList)):
        absTimeList[i] = absTimeList[i-1] + timeList[i]
    return absTimeList

def addIt0(list, value):
    newList = np.zeros(len(list)+1)
    newList[0] = value
    newList[1:] = list
    return newList

def plotLists(Y, X = None, stopIndices = None, labels = None, labelsStop = None, title = None, xlabel = None, ylabel = None, saveName = None, linestyle = None, semilogy = False):
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
        
        os.makedirs(os.path.dirname(savePath(saveName)), exist_ok=True)
        if saveName is not None:
            plt.savefig(savePath(saveName), bbox_inches='tight')