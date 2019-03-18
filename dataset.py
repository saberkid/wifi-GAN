from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch


class CSISet(Dataset):
    def __init__(self, csifile, targetfile):
        self._csi = csifile
        self._target = targetfile
        transform = []
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(transform)


    def __getitem__(self, index):
        csi = self._csi[index]
        target = self._target[index]
        return self.preprocess(csi), target

    def __len__(self):
        return len(self._csi)

    def preprocess(self, data):
        data = data.swapaxes(0, 2)
        data = torch.from_numpy(data)
        min_v = torch.min(data)
        range_v = torch.max(data) - min_v
        if range_v > 0:
            normalised = (data - min_v) / range_v
        else:
            normalised = torch.zeros(data.size())
        return self.transform(normalised)


class CSILoader(DataLoader):
    def __init__(self, dataset, opt):
        super(CSILoader,self).__init__(dataset,batch_size=opt.batch_size, shuffle=True)
