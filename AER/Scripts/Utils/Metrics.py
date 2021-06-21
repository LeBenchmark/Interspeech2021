import numpy as np

def getMetric(target, output, metric="CCC"):
    if metric == "CCC":
        result = CCC(target, output)
    if metric == "MSE":
        result = MSE(target, output)
    if metric == "RMSE":
        result = RMSE(target, output)
    if metric == "Accuracy":
        result = Accuracy(target, output)
    if metric == "UAR":
        from sklearn.metrics import recall_score
        result = recall_score(target, output, average='macro')
    if "UAR-" in metric:
    	from Utils.Funcs import smooth
    	from sklearn.metrics import recall_score
    	output = smooth(output, win=int(metric.split('-')[1]))
    	result = recall_score(target, output, average='macro')
    if metric == "AUC":
        from sklearn.metrics import roc_auc_score
        result = roc_auc_score(target, output)
    if "AUC-" in metric:
        from sklearn.metrics import roc_auc_score
        from Utils.Funcs import smooth
        result = roc_auc_score(target, smooth(output, win=int(metric.split('-')[1])))
    if "SmoothAcc-" in metric:
        from Utils.Funcs import smooth
        result = Accuracy(target, smooth(output, win=int(metric.split('-')[1])))
    return result
    
def CCC(y_true, y_pred):
    """
    Calculate the CCC for two numpy arrays.
    """
    x = y_true
    y = y_pred
    xMean = x.mean()
    yMean = y.mean()
    xyCov = (x * y).mean() - (xMean * yMean)
    # xyCov = ((x-xMean) * (y-yMean)).mean()
    xVar = x.var()
    yVar = y.var()
    return 2 * xyCov / (xVar + yVar + (xMean - yMean) ** 2)

def MSE(y_true, y_pred):
    """
    Calculate the Mean Square Error for two numpy arrays.
    """
    mse = (np.square(y_true - y_pred)).mean(axis=0)
    return mse

def RMSE(y_true, y_pred):
    """
    Calculate the Mean Square Error for two numpy arrays.
    """
    return np.sqrt(MSE(y_true, y_pred))

def perfMeasure(y_actual, y_pred):
    """
    Calculate the confusion matrix for two numpy arrays.
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==-1:
           TN += 1
        if y_pred[i]==-1 and y_actual[i]!=y_pred[i]:
           FN += 1

    return (TP, FP, TN, FN)

def Accuracy(tars, outs):
    corrects = 0
    for i, tar in enumerate(tars):
        if tars[i] == outs[i]: corrects += 1
    acc = corrects / len(tars)
    return acc

def confMatrix(tars, outs, numTars=10):
    matrix = np.zeros((numTars, numTars))
    for i, out in enumerate(outs):
        matrix[tars[i], outs[i]] += 1
    return matrix
