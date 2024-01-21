from sklearn import metrics
l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

print('sklearn accuracy:',metrics.accuracy_score(l1,l2))


def accuracy(y_true,y_pred):
    
    correct_counter = 0
    
    for yt,yp in zip(y_true,y_pred):
        if yt==yp:
            correct_counter += 1
    
    return correct_counter/len(y_true)



print('accuracy_v1:',accuracy(l1,l2))


def true_positive(y_true,y_pred):

    tp = 0

    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 1:
            tp +=1

    return tp

def true_negative(y_true,y_pred):

    tn = 0

    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 0:
            tn +=1

    return tn


def false_positive(y_true,y_pred):

    fp = 0

    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 1:
            fp +=1

    return fp

def false_negative(y_true,y_pred):

    fn = 0

    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 0:
            fn +=1

    return fn

l1 = [0,1,1,1,0,0,0,1] 
l2 = [0,1,0,1,0,1,0,0]
print('tp:',true_positive(l1, l2))
print('fp:',false_positive(l1, l2))
print('fn:',false_negative(l1, l2))
print('tn:',true_negative(l1, l2))

def accuracy_v2(y_true,y_pred):

    tp = true_positive(l1, l2)
    fp = false_positive(l1, l2)
    fn = false_negative(l1, l2)
    tn = true_negative(l1, l2)

    accuracy_score = (tp+tn) / (tp+fp+fn+tn)

    return accuracy_score

print('accuracy_v2:',accuracy_v2(l1,l2))


def precision(y_true, y_pred):

    tp = true_positive(l1, l2)
    fp = false_positive(l1, l2)
    
    precision = tp / (tp+fp)

    return precision

print('precision:',precision(l1,l2))


def recall(y_true, y_pred):

    tp = true_positive(l1, l2)
    fn = false_negative(l1, l2)
    
    recall = tp / (tp+fn)

    return recall

print('recall:',recall(l1,l2))


def f1(y_true,y_pred):

    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)

    score = 2 * p * r /(p+r)

    return score

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

print('f1 score :',f1(y_true,y_pred))
print('sklearn f1 score :',metrics.f1_score(y_true,y_pred))


def tpr(y_true, y_pred):

    tp = true_positive(l1, l2)
    fn = false_negative(l1, l2)
    
    recall = tp / (tp+fn)

    return recall

def fpr(y_true, y_pred):

    fp = false_positive(l1, l2)
    tn = true_negative(l1, l2)
    
    fpr = fp / (fp+tn)

    return fpr


y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2,0.85, 0.15, 0.99]


print('auc value:',metrics.roc_auc_score(y_true,y_pred))





# empty lists to store true positive # and false positive values
tp_list = []
fp_list = []
# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2,0.85, 0.15, 0.99]
# some handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
# loop over all thresholds
for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred] # calculate tp
    temp_tp = true_positive(y_true, temp_pred)
    # calculate fp
    temp_fp = false_positive(y_true, temp_pred)
    # append tp and fp to lists
    tp_list.append(temp_tp)
    fp_list.append(temp_fp)


print(tp_list)
print(fp_list)


import numpy as np

def log_loss(y_true,y_proba):

    epsilon = 1e-15

    loss = []

    for yt,yp in zip(y_true,y_proba):

        yp = np.clip(yp,epsilon,1-epsilon)

        temp_loss = -1.0*(
            yt * np.log(yp) +
            (1-yt) * np.log(1-yp)
        )

        loss.append(temp_loss)
    
    return np.mean(loss)

y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2,0.85, 0.15, 0.99]

print('log loss :',log_loss(y_true,y_proba))
print('sklearn log loss :',metrics.log_loss(y_true,y_proba))




#macro precision

def macro_precision(y_true,y_pred):

    num_classes = len(np.unique(y_true))

    precision = 0

    for class_ in range(num_classes):


        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true,temp_pred)
        fp = false_positive(temp_true,temp_pred)

        temp_precision = tp / (tp+fp)
        precision += temp_precision

    precision /= num_classes

    return precision


#micro precision

def micro_precision(y_true,y_pred):

    num_classes = len(np.unique(y_true))

    tp = 0
    fp = 0

    for class_ in range(num_classes):


        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp += true_positive(temp_true,temp_pred)
        fp += false_positive(temp_true,temp_pred)


    precision = tp / (tp+fp)

    return precision
    
#weighted precision

from collections import Counter

def weighted_precision(y_true,y_pred):

    num_classes = len(np.unique(y_true))

    class_counts = Counter(y_true)

    precision = 0

    for class_ in range(num_classes):


        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true,temp_pred)
        fp = false_positive(temp_true,temp_pred)

        temp_precision = tp / (tp+fp)
        
        weighted_precision = class_counts[class_] * temp_precision

        precision += weighted_precision


    overall_precision = precision/len(y_true)

    return overall_precision



from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
print('macro_precision:',macro_precision(y_true, y_pred))
print('micro_precision:',micro_precision(y_true, y_pred))
print('weighted_precision:',weighted_precision(y_true, y_pred))


# #confusion matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
# y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

# cm = metrics.confusion_matrix(y_true,y_pred)

# plt.figure(figsize = (10,10))

# cmap = sns.cubehelix_palette(50,hue =0.05,rot = 0,light =0.9,dark = 0,as_cmap = True)

# sns.set(font_scale = 2.5)

# sns.heatmap(cm,annot = True,cmap = cmap ,cbar = False)

# plt.ylabel('actual labels', fontsize = 20)
# plt.xlabel('predicted labels', fontsize = 20)
# plt.show()

def pk(y_true , y_pred , k):

    if k == 0:
        return 0

    y_pred = y_pred[:k]

    pred_set  = set(y_pred)

    true_set = set(y_true)

    common_values = pred_set.intersection(true_set)

    return len(common_values) / len(y_pred[:k])



def apk(y_true, y_pred, k):

    
    pk_values = []

    for i in range(1,k+1):
        pk_values.append(pk(y_true, y_pred , i))

    if len(pk_values) == 0:
        return 0

    return sum(pk_values) / len(pk_values)



y_true = [
[1, 2, 3],
[0, 2],
[1],
[2, 3],
[1, 0],
[]
]

y_pred = [
   [0, 1, 2],
    [1],
    [0, 2, 3],
    [2, 3, 4, 0],
    [0, 1, 2],
    [0]
]

for i in range(len(y_true)):
    for j in range(1,4):
        print(
            f"""
            y_true = {y_true[i]},
            y_pred = {y_pred[i]},
            AP@{j} = {apk(y_true[i],y_pred[i],k=j)}
            """
        )


def mapk(y_true, y_pred , k):

    apk_values = []

    for i in range(len(y_true)):

        apk_values.append(apk(y_true[i],y_pred[i], k))

    return sum(apk_values)/len(apk_values)

print('mapk: ')

print('mapk: k=1',mapk(y_true,y_pred , k=1))
print('mapk: k=2',mapk(y_true,y_pred , k=2))
print('mapk: k=3',mapk(y_true,y_pred , k=3))
print('mapk: k=4',mapk(y_true,y_pred , k=4))