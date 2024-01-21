import numpy as np


def mean_absolute_error(y_true, y_pred):
    

    error = 0

    for yt,yp in zip(y_true,y_pred):
        error += np.abs(yt-yp)

    return error / len(y_true)


def mean_squared_error(y_true, y_pred):
    

    error = 0

    for yt,yp in zip(y_true,y_pred):
        error += (yt-yp)**2

    return error / len(y_true)



def mean_squared_log_error(y_true, y_pred):

    error = 0

    for yt,yp in zip(y_true, y_pred):
        error += (np.log(1+yt) - np.log(1+yp))**2

    return error / len(y_true)

def mean_percentage_error(y_true, y_pred):

    error = 0

    for yt,yp in zip(y_true, y_pred):
        error += (yt-yp)/yt
    
    return error / len(y_true)


def mean_abs_percentage_error(y_true, y_pred):

    error = 0

    for yt,yp in zip(y_true, y_pred):
        error += np.abs(yt-yp)/yt
    
    return error / len(y_true)


def r2(y_true, y_pred):

    mean_true_value = np.mean(y_true)

    numerator = 0
    denominator = 0

    for yt, yp in zip(y_true, y_pred):
        numerator += (y_true - y_pred)**2
        denominator += (y_true - mean_true_value)**2
    
    ratio = numerator/denominator

    return 1 - ratio


def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


from sklearn import metrics

y_true = [1,2,3,1,2,3,1,2,3]
y_pred = [2,1,3,1,2,3,3,1,2]

print(metrics.cohen_kappa_score(y_true, y_pred, weights = 'quadratic'))
print(metrics.accuracy_score(y_true,y_pred))


def mcc(y_true, y_pred):

    tp = true_positive(y_true,y_pred)
    tn = true_negative(y_true,y_pred)
    fp = false_positive(y_true,y_pred)
    fn = false_negative(y_true,y_pred)

    numerator = (tp*tn) - (fp*fn)

    denominator = (
        (tp+fp) *
        (fn+tn) *
        (fp+tn) *
        (tp+fn)
    )

    denominator = denominator ** 0.5

    return numerator/denominator

    