import numpy as np
import scipy.signal as signal

from reader import csi_parse


def getCSI(filename):
    return csi_parse.csi_to_data(filename)


def getMagnitude(data0):
    csiData = data0['H']
    RssiData = data0['Rssi']
    return np.abs(csiData).astype(np.float64), np.abs(RssiData).astype(np.float64)

