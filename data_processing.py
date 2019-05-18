import os

from sklearn.decomposition import IncrementalPCA
from utils.emd import emd_csi
from reader import Parser
import numpy as np
import pickle
import joblib
import glob

class_dict_local  = {'empty':0, 'living': 1, 'kitchen':2, 'chianyu':3, 'bathroom':4, 'jen':5}
class_dict_counting  = {'empty':0, 's':0, 'w': 1, 'ss': 0, 'sw':1, 'ww':2, 'ssw':1, 'sww':2, 'www':3}

SUBCARRIER_S = 8
STREAM_S = 4
IMF_S = 8
WINDOW_OVERLAP = 20
WINDOW_SIZE = 100
#============read CSI data from file======================
# subcarrier channel, samples point, antena to antena
def get_csi(filename):
    data0, n_subcarriers, n_transmitters, n_receivers, n_packets = Parser.getCSI(filename)
    # print("NO. Subcarriers: ", n_subcarriers, "NO. transmitters: ", n_transmitters, "NO. receivers: ", n_receivers,
    #       "NO. n_packets: ", n_packets)
    csiAmplitude_raw, rssi_raw = Parser.getMagnitude(data0)
    # print('csiAmplitude_raw: ', csiAmplitude_raw.shape)
    #csiAmplitude_filtered = Parser.getFiltered(csiAmplitude_raw, 'original', 8,
                                               # polyorder=3,
                                               # mode='nearest',
                                               # axis=0)
    return csiAmplitude_raw


def process_local(filepath, trim=800):
    filepath = '../data/32-'

    for dir in os.listdir(filepath):
        if not os.path.isdir(filepath + '/' + dir):
            continue
        csiset = []
        target = []
        for file in os.listdir(filepath + '/' + dir):
            filename = filepath + '/' + dir + '/' + file
            print('processing' + filename)
            csi = get_csi(filename)
            discard = (csi.shape[1] - trim) // 2
            csi = csi[:, discard:discard + trim]
            classnum = class_dict_local[os.path.splitext(file)[0].split('_')[-1]]
            #sliding window
            for i in range(32):
                csiAmplitude_filtered_sub = csi[:, i * 24:i * 24 + 48, ]
                csiset.append(csiAmplitude_filtered_sub)
                target.append(classnum)


        csiset = np.array(csiset)
        target = np.array(target)
        data = {'x': csiset, 'y':target}
        csi_shape = csiset.shape
        print('csi processed: ', csi_shape)
        print(target.shape)
        with open("../data/chianyu_{0}.pkl".format(dir), 'wb+') as f:
            pickle.dump(data, f)


def get_pca_model(filepath):
    joblib_file = filepath + '/' + 'CSI_pca_model'
    if os.path.exists(joblib_file):
        return joblib.load(joblib_file)
    ipca = IncrementalPCA(n_components=SUBCARRIER_S)
    #ipca = IncrementalPCA()
    for dir in os.listdir(filepath):
        csiset = []
        target = []
        try:
            for filename in glob.glob(filepath + '/' + dir + '/*.csi'):
                print('processing' + filename)
                csi = get_csi(filename)
                # #filtered out 16
                csi = csi[:, :, 0: STREAM_S].swapaxes(0, 1) # (t * subcarriers * )
                csi = csi.reshape(csi.shape[0], -1)
                print(csi.shape)
                ipca.partial_fit(csi)
        except NotADirectoryError:
            pass
    print(ipca.explained_variance_ratio_)
    joblib_file = "CSI_pca_model"
    joblib.dump(ipca, filepath + '/' +joblib_file)
    return ipca


def process_count(filepath, ipca):
    for dir in os.listdir(filepath):
        if not os.path.isdir(filepath + '/' + dir):
            continue
        csiset = []
        target = []
        for filename in glob.glob(filepath + '/' + dir + '/*.csi'):
            #filename = filepath + '/' + dir + '/' + file
            label = filename.split('_')[-1][:-4]
            if label in class_dict_counting:
                print('processing' + filename)
                csi = get_csi(filename)

                # filtered out 16 to 4
                csi = csi[:, :, 0:STREAM_S].swapaxes(0, 1)

                # PCA
                print('before' + str(csi.shape))
                csi_pca = np.zeros((csi.shape[0], SUBCARRIER_S, STREAM_S))
                for stream in range(STREAM_S):
                    csi_pca[:, :, stream] = ipca.transform(csi[:, :, stream])
                print('after' + str(csi_pca.shape))

                # EMD
                csi_emd = np.zeros((csi.shape[0], SUBCARRIER_S, STREAM_S, IMF_S))
                for subcarrier in range(csi_pca.shape[1]):
                    for stream in range(csi_pca.shape[2]):
                        csi_sub_emd = emd_csi(csi_pca[:, subcarrier, stream], IMF_S)
                        csi_emd[:, subcarrier, stream, :] = csi_sub_emd
                csi_emd = csi_emd.reshape(csi_emd.shape[0], SUBCARRIER_S, -1)

                # Cut into windows
                csi_windows = cut_into_windows(csi_emd)
                label_num = class_dict_counting[label]
                labels = [label_num] * len(csi_windows)
                csiset.extend(csi_windows)
                target.extend(labels)

        csiset = np.array(csiset)
        target = np.array(target)
        data = {'x': csiset, 'y': target}
        csi_shape = csiset.shape
        print('***********csi processed shape*********', csi_shape)
        print('************target shape*************', target.shape)
        with open("data/counting/counting_{0}.pkl".format(dir), 'wb+') as f:
            pickle.dump(data, f)

def process_count_raw(filepath):
    for dir in os.listdir(filepath):
        if not os.path.isdir(filepath + '/' + dir):
            continue
        csiset = []
        target = []
        for filename in glob.glob(filepath + '/' + dir + '/*.csi'):
            #filename = filepath + '/' + dir + '/' + file
            label = filename.split('_')[-1][:-4]
            if label in class_dict_counting:
                print('processing' + filename)
                csi = get_csi(filename)

                csi = csi.swapaxes(0, 1)

                # Cut into windows
                csi_windows = cut_into_windows(csi)
                label_num = class_dict_counting[label]
                labels = [label_num] * len(csi_windows)
                csiset.extend(csi_windows)
                target.extend(labels)

        csiset = np.array(csiset)
        target = np.array(target)
        data = {'x': csiset, 'y': target}
        csi_shape = csiset.shape
        print('***********csi processed shape*********', csi_shape)
        print('************target shape*************', target.shape)
        with open("data/counting/counting_{0}.raw".format(dir), 'wb+') as f:
            pickle.dump(data, f)


def save_sub_mean(filepath):
    for dir in os.listdir(filepath):
        if not os.path.isdir(filepath + '/' + dir):
            continue
        cur_mean = get_empty_mean(filepath + '/' + dir)
        if cur_mean is not None:
            print(cur_mean.shape)
            for file in os.listdir(filepath + '/' + dir):
                filename = filepath + '/' + dir + '/' + file
                label = filename.split('_')[-1][:-4]
                if label in class_dict_counting and label != 'empty':
                    print('processing' + filename)
                    csi = get_csi(filename)
                    csi = csi.swapaxes(0, 1)
                    csi = csi - cur_mean

                    dumpname = filepath + '/' + dir + '/sub_' + file
                    dumpname = dumpname.replace('.csi', '.pkl')
                    with open(dumpname, 'wb+') as f:
                        pickle.dump(csi, f)


def get_empty_mean(filepath):
    csi = None
    try:
        for file in os.listdir(filepath):
            filename = filepath + '/' + file
            label = filename.split('_')[-1][:-4]
            if label == 'empty':
                print('getting mean for ' + filename)
                csi = get_csi(filename)
                csi = csi.swapaxes(0, 1)
    except NotADirectoryError:
        pass
    return np.mean(csi, axis=0) if csi is not None\
        else None


def cut_into_windows(csi,trim=100):
    csi = csi[trim//2: -trim//2]
    csi_windows = []
    # sliding window
    i = 0
    while i * WINDOW_OVERLAP + WINDOW_SIZE <= csi.shape[0]:
        csi_window = csi[i * WINDOW_OVERLAP:i * WINDOW_OVERLAP + WINDOW_SIZE]
        csi_windows.append(csi_window)
        i += 1
    return csi_windows

ipca = get_pca_model(filepath = 'data/counting')
#ipca = get_pca_model(filepath = '../data/32-')
#process_count(filepath = 'data/counting', ipca=ipca)
#process_count_raw(filepath = 'data/counting')
# save_sub_mean('../data/counting')

