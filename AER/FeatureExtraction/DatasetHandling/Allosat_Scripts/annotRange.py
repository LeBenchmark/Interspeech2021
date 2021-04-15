'''
change emotional ratings range of data!
'''
import os, glob
import pandas as pd
from shutil import copyfile
from scipy import signal
import scipy.interpolate
import pandas as pd
import numpy as np
import math

origFolder = '../Annots/labels_0.01_U'
outFolder = '../Annots/labels_0.01'

if not os.path.exists(outFolder): os.makedirs(outFolder)

filePaths = glob.glob(os.path.join(origFolder, '*'), recursive=True)

for filePath in sorted(filePaths):
    AfterLastSlash = filePath.rfind(os.path.split(filePath)[-1])
    fileName = filePath[AfterLastSlash:]
    new_path = os.path.join(outFolder, fileName)
    df = pd.read_csv(filePath, delimiter=',')
    header = df.keys()
    out = df.to_numpy()[:,:].astype('float64')
    out = np.append(out[0], out).reshape(out.shape[0]+1, out.shape[1]) # ADDING ZERO TIME ---------
    out[0,0] = 0
    # print(oewOut.shape, out.shape, np.arange(0, len(out[:,0]), (newRate/origRate)).shape, np.arange(0, len(out[:,0]), 1).shape, (newRate/origRate))
    newOut = out
    for i in range(out.shape[1]):
        newOut[:, i] = out[:,i] * 2 - 1
    out = newOut
    df = pd.DataFrame(data=out)
    df.to_csv(new_path, header=header, index=False)