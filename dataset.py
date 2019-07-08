from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch


class CSISet(Dataset):
    def __init__(self, csifile, targetfile, imf_s = 6, imf_selection=False):
        self._csi = csifile
        self._target = targetfile
        transform = []
        #transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(transform)
        self.imf_s = imf_s
        self.imf_selection = imf_selection


    def __getitem__(self, index):
        csi = self._csi[index]
        target = self._target[index]
        return self.preprocess(csi), target

    def __len__(self):
        return len(self._csi)

    def preprocess(self, data):
        if self.imf_selection:
            data = data[:, :, 0 : self.imf_s]
            data = data.reshape(data.shape[0], -1)

        data = data.swapaxes(0, 1)
        data = torch.from_numpy(data)
        # min_v = torch.min(data)
        # range_v = torch.max(data) - min_v
        # if range_v > 0:
        #     normalised = (data - min_v) / range_v
        # else:
        #     normalised = torch.zeros(data.size())
        return self.transform(data)


class CSILoader(DataLoader):
    def __init__(self, dataset, opt, sampler=None, shuffle=False):
        super(CSILoader,self).__init__(dataset,batch_size=opt.batch_size, sampler=sampler, shuffle=shuffle)
