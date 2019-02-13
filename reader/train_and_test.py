
import torch.utils.data as data
import torch
from core.wavelets import spectro_wifi
import numpy.core
import numpy as np
import os
import os.path
import glob
import pickle


def compile2d(spectrogramDict, trim=800):
    dataY_train = []
    dataY_test = []
    dataY = []
    dataX = []
    dataX_train = []
    dataX_test = []
    data_train = []
    data_test = []
    samples_train = []
    samples_test = []
    index = 0

    for areaId in CHclasses:

        # Shape: n_samples x n_packets x n_frequencies
        samples = spectrogramDict[areaId]

        if trim > 0:
            discard = (len(samples) - trim) // 2
            samples = samples[discard:discard + trim, ...]
            for i in range(600):
                samples_train.append([samples[i], index])
            for j in range(200):
                samples_test.append([samples[j+600], index])


            #samples_train = samples[0:600, ...]
            #samples_test = samples[600:800, ...]
        index += 1
        #dataX.append(samples)
        #dataX_train.append(samples_train)
        #dataX_test.append(samples_test)


        # dataY.append([index] * samples.shape[0])
        #
        # dataY_train.append([index] * 600)
        # dataY_test.append([index] * 200)


        # data_train.append([dataX_train, dataY_train])
        # data_test.append([dataX_test, dataY_test])

    # dataY = to_categorical(dataY, num_classes=len(classList))

    return samples_train, samples_test


CHclasses = ['stove', 'dining', 'computer', 'couch']
fr = open('/Users/landu/Documents/Aerial Project/data/cnn_test/SpChianyuAptDay1-round32-S16-S58.pkl', 'rb')
data_loc = pickle.load(fr)

fr.close()

train_box = []
test_box = []
train_data, test_data = compile2d(data_loc, 800)



fw_train = open('/Users/landu/Documents/Aerial Project/data/train_test/SpChianyuAptDay1-train-round32-S16-S58.pkl', 'wb')
fw_test = open('/Users/landu/Documents/Aerial Project/data/train_test/SpChianyuAptDay1-test-round32-S16-S58.pkl', 'wb')

print('data shape: ', len(train_data), ' : ', len(test_data))

pickle.dump(train_data, fw_train, -1)
pickle.dump(test_data, fw_test, -1)

fw_train.close()
fw_test.close()







# def compile2d(self, spectrogramDict, trim=800):
#     dataX = None
#     dataY = []
#
#     index = 0
#
#     for areaId in CHclasses:
#
#         # Shape: n_samples x n_packets x n_frequencies
#         samples = spectrogramDict[areaId]
#
#         if trim > 0:
#             discard = (len(samples) - trim) // 2
#             samples = samples[discard:discard + trim, ...]
#
#         if dataX is None:
#             dataX = samples
#         else:
#             dataX = np.append(dataX, samples, axis=0)
#         dataY.extend([index] * samples.shape[0])
#
#         index += 1
#
#     # dataX = FeatureExtraction.extractFeatures(dataX, featureSetId=featureSetId)
#     dataX = np.array(dataX)
#     dataY = np.array(dataY)
#     # dataY = to_categorical(dataY, num_classes=len(classList))
#
#     return dataX, dataY