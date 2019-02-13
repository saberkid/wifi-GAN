from torch.utils.data import Dataset, DataLoader

import numpy as np

csifile = np.load('csiset.npy')
targetfile = np.load('target.npy')

def preprocess(csiset):
    # TODO train test split here
    return csiset


def default_loader(csiset):
    csi_tensor = preprocess(csiset)
    return csi_tensor


class CSISet(Dataset):
    def __init__(self, loader=default_loader):
        self._csi = csifile
        self._target = targetfile
        self.loader = loader

    def __getitem__(self, index):
        fn = self._csi[index]
        csi = self.loader(fn)
        target = self._target[index]
        return csi,target

    def __len__(self):
        return len(self._csi)


class CSILoader(DataLoader):
    def __init__(self, dataset):
        super(CSILoader,self).__init__(dataset)
