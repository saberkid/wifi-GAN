import os

from reader import Parser
import numpy as np
import pickle

class_dict  = {'empty':0, 'living': 1, 'kitchen':2, 'chianyu':3, 'bathroom':4, 'jen':5}

#============read CSI data from file======================
# subcarrier channel, samples point, antena to antena
filepath = '../data/32-'

trim = 800

for dir in os.listdir(filepath):
    csiset = []
    target = []
    for file in os.listdir(filepath + '/' + dir):
        filename = filepath + '/' + dir + '/' + file
        print(filename)
        data0, n_subcarriers, n_transmitters, n_receivers, n_packets  = Parser.getCSI(filename)
        # print("NO. Subcarriers: ", n_subcarriers, "NO. transmitters: ", n_transmitters, "NO. receivers: ", n_receivers,
        #       "NO. n_packets: ", n_packets)
        csiAmplitude_raw, rssi_raw = Parser.getMagnitude(data0)
        # print('csiAmplitude_raw: ', csiAmplitude_raw.shape)
        #csiAmplitude = Parser.getStandardized(csiAmplitude_raw)

        csiAmplitude_filtered = Parser.getFiltered(csiAmplitude_raw, 'original', 8,
                                                   polyorder=3,
                                                   mode='nearest',
                                                   axis=0)
        # #filtered out 16
        # csiAmplitude_filtered = csiAmplitude_filtered[ :, :, 0]

        discard = ( csiAmplitude_filtered.shape[1] - trim) // 2
        csiAmplitude_filtered = csiAmplitude_filtered[:, discard:discard + trim]
        classnum = class_dict[os.path.splitext(file)[0].split('_')[-1]]
        #sliding window
        for i in range(32):
            csiAmplitude_filtered_sub = csiAmplitude_filtered[:, i * 24:i * 24 + 48, ]
            csiset.append(csiAmplitude_filtered_sub)
            target.append(classnum)

    csiset = np.array(csiset)
    target = np.array(target)
    data = {'x': csiset, 'y':target}
    csi_shape = csiset.shape
    print('csiAmplitude_filtered: ', csi_shape)
    print(target.shape)
    with open("../data/chianyu_{0}.pkl".format(dir), 'wb+') as f:
        pickle.dump(data, f)

######################################################################
#L2_csi, L2_rssi, L2_ntx = ParserTools.get_L2_NormsFiltered_data_adaptive()                                                                                     files)
#best_stream, best_8_streams, L2_norm_means = csi_utils.get_best_stream(L2_NormsFiltered_data, L2_NormsFiltered_Rssi, t_window, window_size80)
#=====================================================
# n_components = 5
# magnitude = csiAmplitude_filtered
# X_train_01 = magnitude[:,:,1]
# X_train_02 = magnitude[:,:,2]
# X_train_03 = magnitude[19,:,:]
# X_train_04 = magnitude[26,:,:]
# X_train_05 = magnitude[33,:,:]
# X_train_06 = magnitude[40,:,:]
# X_train_07 = magnitude[47,:,:]
# X_train_08 = magnitude[54,:,:]
# #pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
#
# data = X_train_02
# data_T = np.transpose(data)
# print(data_T.shape)
#
# pca = decomposition.PCA(n_components=10)
# pca_data = pca.fit_transform(data_T)
#
# print(pca_data.shape)
# print(pca.explained_variance_ratio_)
#
# sampling_rate = 20
# t = np.arange(0, data_T.shape[0], 1.0)
#
# wavename = 'cgau8'
# scaless = np.arange(1, 512)
# cwtmatrt_p1, freqs_p1 = pywt.cwt(pca_data[:,0], scaless, wavename, sampling_period=5)
# cwtmatrt_p2, freqs_p2 = pywt.cwt(pca_data[:,1], scaless, wavename, sampling_period=5)
# cwtmatrt_p3, freqs_p3 = pywt.cwt(pca_data[:,2], scaless, wavename, sampling_period=5)
#
# print('cwtmatrt: ', cwtmatrt_p1.shape, 'Frequency: ', freqs_p1.shape)
#
# plt.figure(figsize=(8, 4))
# plt.subplot(311)
# plt.contourf(t,freqs_p1,abs(cwtmatrt_p1))
# plt.ylabel(u"Frequency(Hz)")
# plt.xlabel(u"Time(S)")
# plt.subplots_adjust(hspace=0.4)
# plt.subplot(312)
# plt.contourf(t,freqs_p2,abs(cwtmatrt_p2))
# plt.ylabel(u"Frequency(Hz)")
# plt.xlabel(u"Time(S)")
# plt.subplots_adjust(hspace=0.4)
# plt.subplot(313)
# plt.contourf(t,freqs_p3,abs(cwtmatrt_p3))
# plt.ylabel(u"Frequency(Hz)")
# plt.xlabel(u"Time(S)")
# plt.subplots_adjust(hspace=0.4)
# plt.show()
#print(X_train.shape)