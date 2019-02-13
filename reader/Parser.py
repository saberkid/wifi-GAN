import numpy as np
import scipy.signal as signal

from reader import ParserTools


def getCSI(filename):
    return ParserTools.csi_to_data(filename)


def getMagnitude(data0):
    csiData = data0['H']
    RssiData = data0['Rssi']
    return np.abs(csiData), np.abs(RssiData)


def getStandardized(magnitude):
    return ParserTools.get_normalized_data(magnitude)



def getFiltered(magnitude, filterName, windSize, **kwargs):
    if windSize % 2 == 0:
        windSize += 1
        print('Warning: Window size cannot be an even number. Value reset to %d' % windSize)

    if filterName == 'original' or filterName == 'legacy':
        n_subcarriers, n_packets, n_streams = magnitude.shape
        return ParserTools.get_Filtered2D_data(magnitude, n_subcarriers, n_packets, n_streams)

    elif filterName == 'median':
        return signal.medfilt(magnitude, kernel_size=windSize)

    elif filterName == 'wiener':
        return signal.wiener(magnitude, mysize=windSize, **kwargs)

    elif filterName == 'savgol':
        return signal.savgol_filter(magnitude, window_length=windSize, **kwargs)

    elif filterName == 'hilbert':
        return np.abs(signal.hilbert(magnitude, **kwargs))

    elif filterName == 'threshold':
        threshold = kwargs.get('threshold', np.mean(magnitude))
        filterFlags1 = magnitude < threshold
        # filterFlags2 = amplitude > -threshold
        # filterFlags = filterFlags1 and filterFlags2
        magnitude[filterFlags1] = threshold
        return magnitude



