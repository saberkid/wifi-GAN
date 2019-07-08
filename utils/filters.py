from scipy import signal
import math
import pywt
import numpy as np


def butterworth(x, fs = 1000, fc=30):
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    x2 = signal.filtfilt(b, a, x)
    return x2

def dwt(x):
    coeffs = pywt.wavedec(x, 'db6', level=3)
    threshold = np.median(coeffs[0]) / 0.6745

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * math.sqrt(2 * math.log(len(x))))
    x2 = pywt.waverec(coeffs, 'db6')
    return x2

# python
# 1D hampel filter
#
# purpose
#  Outlier detection and remove


def hampel(x, k=20, method="center", thr=3):
    # Input
    # x       input data
    # k       half window size (full 2*k+1)
    # mode    about both ends
    #         str {‘center’, 'same','ignore',‘nan’}, optional
    #
    #           center  set center of window at target value
    #           same    always same window size
    #           ignore  set original data
    #           nan     set non
    #
    # thr     threshold (defaut 3), optional
    # Output
    # newX    filtered data
    # omadIdx indices of outliers
    arraySize = len(x)
    idx = np.arange(arraySize)
    newX = x.copy()
    omadIdx = np.zeros_like(x)
    for i in range(arraySize):
        mask1 = np.where(idx >= (idx[i] - k), True, False)
        mask2 = np.where(idx <= (idx[i] + k), True, False)
        kernel = np.logical_and(mask1, mask2)
        if method == "same":
            if i < (k):
                kernel = np.zeros_like(x).astype(bool)
                kernel[:(2 * k + 1)] = True
            elif i >= (len(x) - k):
                kernel = np.zeros_like(x).astype(bool)
                kernel[-(2 * k + 1):] = True
        # print (kernel.astype(int))
        # print (x[kernel])
        med0 = np.median(x[kernel])
        # print (med0)
        s0 = 1.4826 * np.median(np.abs(x[kernel] - med0))
        if np.abs(x[i] - med0) > thr * s0:
            omadIdx[i] = 1
            newX[i] = med0

    if method == "nan":
        newX[:k] = np.nan
        newX[-k:] = np.nan
        omadIdx[:k] = 0
        omadIdx[-k:] = 0
    elif method == "ignore":
        newX[:k] = x[:k]
        newX[-k:] = x[-k:]
        omadIdx[:k] = 0
        omadIdx[-k:] = 0

    return newX#, omadIdx.astype(bool)