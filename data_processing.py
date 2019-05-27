import os

from sklearn.decomposition import IncrementalPCA, PCA
from utils.emd import emd_csi
from reader import Parser
import numpy as np
import pickle
import joblib
import glob
import matplotlib.pyplot as plt

class_dict_local  = {'empty':0, 'living': 1, 'kitchen':2, 'chianyu':3, 'bathroom':4, 'jen':5}
class_dict_counting  = {'empty':0, 's':0, 'w': 1, 'ss': 0, 'sw':1, 'ww':2, 'ssw':1, 'sww':2, 'www':3}

SUBCARRIER_S = 8
STREAM_S = 4
IMF_S = 10
WINDOW_STRIDE = 20
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
    ipca = IncrementalPCA(n_components=SUBCARRIER_S + 1)
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
                csi = csi.reshape(csi.shape[0], -1)
                # PCA
                print('before' + str(csi.shape))
                ipca.n_components = SUBCARRIER_S
                # Discard the first Principal component
                csi_pca = ipca.transform(csi)[:, 1:]
                print('after' + str(csi_pca.shape))

                # Cut into windows
                csi_windows = cut_into_windows(csi_pca)

                # EMD
                for i, csi_pca in enumerate(csi_windows):
                    csi_emd = np.zeros((csi_pca.shape[0], csi_pca.shape[1], IMF_S))
                    for stream in range(csi_pca.shape[1]):
                            csi_sub_emd = emd_csi(csi_pca[:, stream], IMF_S)
                            csi_emd[:, stream, :] = csi_sub_emd
                    #csi_windows[i] = csi_emd.reshape(csi_emd.shape[0], -1)
                    csi_windows[i] = csi_emd


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
    while i * WINDOW_STRIDE + WINDOW_SIZE <= csi.shape[0]:
        csi_window = csi[i * WINDOW_STRIDE:i * WINDOW_STRIDE + WINDOW_SIZE]
        csi_windows.append(csi_window)
        i += 1
    return csi_windows

#ipca = get_pca_model(filepath = 'data/counting')
#ipca = get_pca_model(filepath = '../data/32-')
#process_count(filepath = 'data/counting', ipca=ipca)
#process_count_raw(filepath = 'data/counting')
# save_sub_mean('../data/counting')


filepath = 'data/falldata'
joblib_file = filepath + '/' + 'CSI_pca_model'

#ipca = IncrementalPCA(n_components=SUBCARRIER_S + 1)
ipca = IncrementalPCA(n_components=3)
for data_file in glob.glob(r'{}/*.pkl'.format(filepath)):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        csi = data[0]
        csi = csi.reshape(csi.shape[0], -1)
        ipca.partial_fit(csi)
        #print(ipca.explained_variance_ratio_)

# joblib_file = "CSI_pca_model"
# joblib.dump(ipca, filepath + '/' +joblib_file)

for data_file in glob.glob(r'{}/*.pkl'.format(filepath)):
    with open(data_file, 'rb') as f:
        label = os.path.splitext(data_file)[0].split('_')[-1]
        data = pickle.load(f)
        csi_list  = []
        for csi in data:
            csi = csi.reshape(csi.shape[0], -1)
            csi = ipca.transform(csi)
            csi_list.append(csi)
        csi_list = np.array(csi_list)

        dumpname = data_file.replace('pkl', 'pkls')
        with open(dumpname, 'wb+') as f2:
            pickle.dump(csi_list, f2)

    # for i in range(3):
    #     plt.subplot(3, 1, i+1)
    #     plt.gca().set_title('{} PC{}'.format(label, i+1))
    #     plt.xlabel('Packet Index')
    #     plt.ylabel('CSI Amplitude')
    #     plt.plot(csi[:, i], color='b')
    #
    # plt.show()



