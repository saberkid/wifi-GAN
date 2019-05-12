from __future__ import division, print_function

import numpy  as np
import pylab as plt
from PyEMD import EMD


def emd_csi(csi, imfn=6):
    # Execute EMD on signal
    emd = EMD()
    emd.FIXE = 10
    IMF = emd(csi)

    N = IMF.shape[0]+1
    csi_emd = np.zeros((csi.shape[0], 6))

    for n, imf in enumerate(IMF):
        if 1 <= n <= imfn:
            csi_emd[:, n-1] = imf
    return csi_emd