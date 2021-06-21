def get_files_in_path(path, ext="wav"):
    """
    Get files in a path
    exampe : files = get_files_in_path("./audioFiles")
    """
    import os, glob
    path = os.path.join(path, "*."+ext)
    theFiles = glob.glob(path, recursive=True)
    return theFiles

def find_last_slash_pos_in_path(path):
    """
    Find last slash position in a path
    exampe : files = find_last_slash_pos_in_path("./audioFiles/abc.wav")
    output : integer
        the value that is the position of the last slash
    """
    import os
    LastSlashPos = path.rfind(os.path.split(path)[-1]) - 1
    return LastSlashPos

def search_csv(csv_file, search_term, colomn_searched, colomn_out):
    '''
    Search a string in a csv file and a colomn and get it's corresponding value for a different colomn. 
    example : valenz = search_csv('labels-sorted.csv', '001_01.wav', 'Laufnummer', 'Valenz')
    '''
    import pandas as pd
    df = pd.read_csv(csv_file)
    out = df[df[colomn_searched] == search_term][colomn_out]
    ret = out.values
    if len(ret) == 1:
        return ret[0]
    else:
        return -1

def writeLineToCSV(csvPath, headers, values):
    '''
    Write one line to CSV
    example : writeLineToCSV("test.csv", ["a", "b", "c"], ["something",16,34])
    '''
    import pandas as pd
    import os
    LastSlashPos = csvPath.rfind(os.path.split(csvPath)[-1]) - 1
    if not os.path.exists(csvPath[:LastSlashPos]): os.makedirs(csvPath[:LastSlashPos])
    dic = {}
    for i, header in enumerate(headers): dic[header] = values[i]
    data = [dic]
    if os.path.exists(csvPath): 
        df = pd.read_csv(csvPath)
        df = df.append(data, ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(data, columns = headers)
    df.to_csv(csvPath, index=False)

def arff2csv(arff_path, csv_path=None, _encoding='utf8'):
    """
    This function was copied from https://github.com/Hutdris/arff2csv/blob/master/arff2csv.py
    It turns .arff files into csvs.
    """
    with open(arff_path, 'r', encoding=_encoding) as fr:
        attributes = []
        if csv_path is None:
            csv_path = arff_path[:-4] + 'csv'  # *.arff -> *.csv
        write_sw = False
        with open(csv_path, 'w', encoding=_encoding) as fw:
            for line in fr.readlines():
                if write_sw:
                    if line == "": print("emp")
                    fw.write(line)
                elif '@data' in line:
                    fw.write(','.join(attributes) + '\n')
                    write_sw = True
                elif '@attribute' in line:
                    attributes.append(line.split()[1])  # @attribute attribute_tag numeric
    print("Convert {} to {}.".format(arff_path, csv_path))

def divide_list(list, perc=0.5):
    """
    Divide a list into two new lists. perc is the first list's share. If perc=0.6 then the first new list will have 60 percent of the original list.
    example : f,s = divide_list([1,2,3,4,5,6,7], perc=0.7)
    """
    origLen = len(list)
    lim = int(perc*origLen)
    firstList = list[:lim]
    secondList = list[lim:]
    return firstList, secondList

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = "fit", fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    import os
    rows, columns = os.popen('stty size', 'r').read().split()
    if length=="fit": length = int(columns) // 2
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def csvReader(fullPath, headers, standardize=False):
    import pandas as pd
    import numpy as np
    # print(fullPath)
    df = pd.read_csv(fullPath)
    outs = []
    for header in headers:
        out = df[header].to_numpy()
        if standardize: out = (out - out.mean(axis=0)) / out.std(axis=0)
        out = np.expand_dims(out, axis=1)
        outs.append(out)
    outs = np.concatenate(outs, 1)
    return outs

def loadFromJson(jsonPath):
    import json
    jsonDict = {}
    with open(jsonPath, 'r') as jsonFile: 
        jsonDict = json.load(jsonFile)
    return jsonDict

def smooth(sig, win=25*1):
    import numpy as np
    mysig = sig.copy()
    aux = int(win/2)
    for i in range(aux, len(mysig)-aux):
        value = np.mean(sig[i-aux:i+aux])
        mysig[i] = 1 if value > 0 else -1
    mysig[:aux] = 1 if np.mean(sig[:aux]) > 0 else -1
    mysig[-aux:] = 1 if np.mean(sig[-aux:]) > 0 else -1
    return mysig

def hysteresis(sig, win=25*1, bottom=-0.8, top=0):
    import numpy as np
    mysig = sig.copy()
    aux = int(win/2)
    mysig[0] = 1 if mysig[0] > top else -1
    for i in range(1, len(mysig)):
        if mysig[i] >= top:
            mysig[i] = 1
        if mysig[i] >= bottom and mysig[i] < top:
            if mysig[i-1] == 1: 
                mysig[i] = 1
            else:
                mysig[i] = -1
        if mysig[i] < bottom:
            mysig[i] = -1
    start = 0
    for i in range(1, len(mysig)):
        if mysig[i] == 1 and mysig[i-1] == -1: start = i
        if mysig[i] == -1 and mysig[i-1] == 1: 
            if i-start < win: 
                for j in range(start, i):
                    mysig[j] = -1 
    return mysig
    
def CCC(y_true, y_pred):
    """
    Calculate the CCC for two numpy arrays.
    """
    import numpy as np
    x = y_true
    y = y_pred
    xMean = x.mean()
    yMean = y.mean()
    xyCov = (x * y).mean() - (xMean * yMean)
    xVar = x.var()
    yVar = y.var()
    return 2 * xyCov / (xVar + yVar + (xMean - yMean) ** 2)

def read_wave_file(wavPath):
    from pydub import AudioSegment
    audio_file = AudioSegment.from_wav(wavPath)
    return audio_file.get_array_of_samples()

def reshapeMatrix(mat, newLen, axis=0):
    import numpy as np
    oldLen = np.shape(mat)[axis]
    newOut = np.zeros((len(np.arange(0, len(mat[:,0]), (oldLen/newLen))), mat.shape[1]), dtype=float)
    dim = mat.shape[1] if axis == 0 else mat.shape[0]
    for i in range(dim):
        newOut[:, i] = np.interp(np.arange(0, len(mat[:,i]), (oldLen/newLen)), np.arange(0, len(mat[:,i]), 1), mat[:,i])
    return newOut
