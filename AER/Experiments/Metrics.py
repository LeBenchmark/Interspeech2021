import numpy as np

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
