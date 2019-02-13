import multiprocessing
import os
import pickle
import re

import numpy as np

import Hyperparameters
from test import Parser


def getFilename(csiType, csiDay, csiRound):
    # E.g. AmChianyuApt1Day1-round01-W50-S58-S16
    filename = '%s%s-%s-W%02d-S%02d-S%02d' % (csiType, csiDay, csiRound,
                                              Hyperparameters.WINDOW_SIZE,
                                              Hyperparameters.STR_SELECTION,
                                              Hyperparameters.SUB_SELECTION,
                                              )
    return filename


def readData(csiDir, filename, inclParams=False):
    with open('%s/%s.pkl' % (csiDir, filename), 'rb') as f:
        data = pickle.load(f)
    return data


def saveData(csiDir, filename, data):
    if not os.path.isdir(csiDir):
        os.makedirs(csiDir)

    with open('%s/%s.pkl' % (csiDir, filename), 'wb') as f:
        pickle.dump(data, f)


def getLabel(filename):
    patt = '\.|_'
    return re.split(patt, filename)[-2]


def selectMagnitude(magnitude):
    streams, subcarriers = Hyperparameters.getSelections()
    streamsMi, subcarriersMi = np.array(streams), np.array(subcarriers)

    magnitudeCo = np.stack((
        magnitude[:, streamsMi[:, None], np.array([j - 3 for j in subcarriers])],
        magnitude[:, streamsMi[:, None], np.array([j - 2 for j in subcarriers])],
        magnitude[:, streamsMi[:, None], np.array([j - 1 for j in subcarriers])],
        magnitude[:, streamsMi[:, None], subcarriersMi],
        magnitude[:, streamsMi[:, None], np.array([j + 1 for j in subcarriers])],
        magnitude[:, streamsMi[:, None], np.array([j + 2 for j in subcarriers])],
        magnitude[:, streamsMi[:, None], np.array([j + 3 for j in subcarriers])],
    ), axis=3)

    return np.median(magnitudeCo, axis=3)


# return np.average(magnitudeCo, axis=3,
# weights=[0.06136, 0.24477, 0.38774, 0.24477, 0.06136],
# weights=[0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598],
# )
# return magnitudeMi


def shapeData(csiData):
    def sliding_window_h(a, windSize, stepSize):
        return np.hstack(a[i:1 + i - windSize or None:stepSize] for i in range(0, windSize))

    if len(csiData.shape) == 1:
        csiData = csiData[np.newaxis, ..., np.newaxis]
    elif len(csiData.shape) == 2:
        csiData = csiData[..., np.newaxis]
    # csiData = np.transpose(csiData, (1, 0, 2))
    csiData = np.expand_dims(csiData, 1)
    csiData_frameArr = sliding_window_h(csiData,
                                        Hyperparameters.WINDOW_SIZE,
                                        Hyperparameters.WINDOW_STEP)
    return csiData_frameArr


def compileData_async(areaPath, key):
    # Part 1 ---------------------------------------------------------------------------------------------------
    # Amplitude is n_subcarriers x n_packets x n_streams
    # Standardize and filter along frequency axis (subcarriers)
    data0, n_subcarriers, n_transmitters, n_receivers, n_packets = Parser.getCSI(areaPath)
    csiAmplitude_raw = Parser.getMagnitude(data0)
    csiAmplitude_normalized = Parser.getStandardized(csiAmplitude_raw)
    csiAmplitude_filtered = Parser.getFiltered(csiAmplitude_normalized, 'savgol', 7,
                                               polyorder=3,
                                               mode='nearest',
                                               axis=0)
    csiAmplitude_transposed = np.transpose(csiAmplitude_filtered, (1, 2, 0))
    csiAmplitude_part1 = csiAmplitude_transposed
    # print('Part 1', csiAmplitude_part1.shape)

    # Part 2 ---------------------------------------------------------------------------------------------------
    csiAmplitude_selected = selectMagnitude(csiAmplitude_part1)
    csiAmplitude_shaped = shapeData(csiAmplitude_selected)
    csiAmplitude_part2 = csiAmplitude_shaped
    # print('Part 2', csiAmplitude_part2.shape)

    return key, csiAmplitude_part2


captureDict = {}


def compileData_callback(res):
    global captureDict

    key, val = res
    if key in captureDict:
        captureDict[key].append(val)
    else:
        captureDict[key] = [val]


# args: specifies the roundId which system compiles
def compileData(csiDir, *args):
    global captureDict
    captureDict = {}
    pool = multiprocessing.Pool()

    roundList = os.listdir(csiDir)
    for roundId in roundList:

        if args is None or roundId in args:
            pass
        else:
            continue

        roundPath = '%s/%s' % (csiDir, roundId)
        # print('Reading %s...' % roundId)

        areaList = os.listdir(roundPath)
        for areaId in areaList:
            areaPath = '%s/%s' % (roundPath, areaId)

            label = getLabel(areaId)
            print('\t%s' % label)

            pool.apply_async(compileData_async,
                             args=(
                                 areaPath,
                                 label
                             ),
                             callback=compileData_callback)

    pool.close()
    pool.join()

    return captureDict
