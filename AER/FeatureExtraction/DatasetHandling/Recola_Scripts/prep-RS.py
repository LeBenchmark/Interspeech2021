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

origFolder = '../Annots/gs_valence_0.01'
outFolder = '../Annots/gs_valence_0.02'
origRate = 0.01
newRate = 0.02

if not os.path.exists(outFolder): os.makedirs(outFolder)

arousalPaths = glob.glob(os.path.join(origFolder, '*'), recursive=True)

# Resample arousal files
for filePath in sorted(arousalPaths):
    AfterLastSlash = filePath.rfind(os.path.split(filePath)[-1])
    fileName = filePath[AfterLastSlash:]
    new_path = os.path.join(outFolder, fileName)
    df = pd.read_csv(filePath, delimiter=',')
    header = df.keys()
    out = df.to_numpy()[:,0:].astype('float64')
    newOut = np.zeros((len(np.arange(0, len(out[:,0]), (newRate/origRate))), out.shape[1]), dtype=float)
    for i in range(out.shape[1]):
        newOut[:, i] = np.interp(np.arange(0, len(out[:,i]), (newRate/origRate)), np.arange(0, len(out[:,i]), 1), out[:,i])
    out = newOut
    df = pd.DataFrame(data=out)
    df.to_csv(new_path, header=header[0:], index=False)
