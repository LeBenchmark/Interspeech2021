'''
Get the emotional ratings and resample them!
'''

import os, glob
import pandas as pd
from shutil import copyfile
from scipy import signal
import scipy.interpolate
import pandas as pd
import numpy as np
import math

origFolder = '../Annots/labels_0.25'
outFolder = '../Annots/labels_0.02'
origRate = 0.25
newRate = 0.02

if not os.path.exists(outFolder): os.makedirs(outFolder)

arousalPaths = glob.glob(os.path.join(origFolder, '*'), recursive=True)

# Resample arousal files
for filePath in sorted(arousalPaths):
    AfterLastSlash = filePath.rfind(os.path.split(filePath)[-1])
    fileName = filePath[AfterLastSlash:]
    new_path = os.path.join(outFolder, fileName)
    df = pd.read_csv(filePath, delimiter=';')
    header = df.keys()
    out = df.to_numpy()[:,1:].astype('float64')
    out = np.append(out[0], out).reshape(out.shape[0]+1, out.shape[1]) # ADDING ZERO TIME ---------
    out[0,0] = 0
    newOut = np.zeros((len(np.arange(0, len(out[:,0]), (newRate/origRate))), out.shape[1]), dtype=float)
    for i in range(out.shape[1]):
        newOut[:, i] = np.interp(np.arange(0, len(out[:,i]), (newRate/origRate)), np.arange(0, len(out[:,i]), 1), out[:,i])
    out = newOut
    df = pd.DataFrame(data=out)
    df.to_csv(new_path, header=header[1:], index=False)

