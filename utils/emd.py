from __future__ import division, print_function

import numpy  as np
import pylab as plt
from PyEMD import EMD, CEEMDAN

def emd_csi(csi, imfn=10):
    # Execute EMD on signal
    emd = EMD()
    emd.FIXE = 10
    IMF = emd(csi)

    csi_emd = np.zeros((csi.shape[0], imfn))
    for n, imf in enumerate(IMF):
        if n < imfn:
            csi_emd[:, n-1] = imf
    return csi_emd
