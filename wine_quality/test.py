# %%
import pandas as pd
from autograd import numpy as np
from autograd import grad
from sklearn.preprocessing import StandardScaler
#C:\Users\USER\Desktop\leedawon\TIL_github\kaggle\wine_quality\test.py

# %%
datas = pd.read_csv("wine_quality/winequality-red.csv")
#print(f"datas head : {datas.head()}")

# %%
#print(f"datas information : {datas.info()}")

# %% [markdown]
# multiple logistic regression will be good!!

# %%
y_col = "quality"
x_cols = datas.columns.drop(["quality"])
#print(x_cols)


# %%
#print(datas[y_col].describe())

# %% [markdown]
# 0-5 quality means 0
# 6-10 quality means 1

# %%
for i in range(len(datas[y_col])):
    if datas[y_col][i] < 6:
        datas[y_col][i] = 0.0
    else:
        datas[y_col][i] = 1.0

# %% [markdown]
# make functions

# %%
def logistic(x):
    out = 1. / (1. + np.exp(-x))
    return out

# %%
def logistic_model(x,params):
    out = logistic(np.dot(x,params[0])+params[1])
    return out

# %%
def model_loss(x,true_labels,params,_lambda=1.0):
    pred = logistic_model(x,params)

    loss =  - (np.dot(true_labels, np.log(pred+1e-15)) + np.dot(1.-true_labels, np.log(1.-pred+1e-15))) \
        + _lambda * np.sum(params[0]**2)
    return loss

# %% [markdown]
# normalize x datas by z-score using sklearn module

# %%
#print(datas[x_cols].describe().loc[["min","max"]])
#print(datas[x_cols].head())

# %%
datas[x_cols] = StandardScaler().fit_transform(datas[x_cols])

# %%
#print(datas[x_cols].describe().loc[["min","max"]])
#print(datas[x_cols].head())

# %%
num_of_datas = datas.shape[0]
num_of_features = len(x_cols)
#print(num_of_datas,num_of_features)

# %%
num_of_val = int(num_of_datas * 0.2)

num_of_test = int(num_of_datas * 0.2)

num_of_train = num_of_datas - num_of_val - num_of_test
#print(f"number of validation : {num_of_val}")
#print(f"number of testing : {num_of_test}")
#print(f"number of trainning : {num_of_train}")
#print(f"sum : {num_of_train + num_of_val + num_of_test}")

# %%
datas_division = np.split(datas[x_cols],[num_of_val, num_of_val + num_of_test],0)
datas_val = datas_division[0]
datas_test = datas_division[1]
datas_train = datas_division[2]

labels_division = np.split(datas[y_col],[num_of_val, num_of_val + num_of_test],0)
labels_val = labels_division[0]
labels_test = labels_division[1]
labels_train = labels_division[2]

# %%
assert labels_train.shape[0] == datas_train.shape[0]
assert labels_val.shape[0] == datas_val.shape[0]
assert labels_test.shape[0] == datas_test.shape[0]

# %%
def classify(x,params):
    probabilities = logistic_model(x,params)
    labels = (probabilities >= 0.5).astype(float)
    return labels

# %%
def performance(predictions, answers, beta = 1.0):
    true_idx = (answers == 1)
    false_idx = (answers == 0)

    n_tp = np.count_nonzero(predictions[true_idx] == 1)

    n_fp = np.count_nonzero(predictions[false_idx] == 1)

    n_tn = np.count_nonzero(predictions[false_idx] == 0)

    n_fn = np.count_nonzero(predictions[true_idx] == 0)

    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)

    score = ((1.0 + beta**2) * precision * recall / (beta ** 2 * precision + recall))

    accuracy = (n_tp + n_tn) / (n_tp + n_fp + n_tn + n_fn)

    return precision,recall,score,accuracy

# %%
gradients = grad(model_loss,argnum = 2)

w = np.zeros(datas_train.shape[1],dtype=float)

b=0.

# %%
pred_labels_test = classify(datas_test,(w,b))
perf = performance(pred_labels_test,labels_test)

#print("Initial precision: {:.1f}%".format(perf[0]*100))
#print("Initial recall: {:.1f}%".format(perf[1]*100))
#print("Initial F-score: {:.1f}%".format(perf[2]*100))
#print("Initial Accuracy: {:.1f}%".format(perf[3]*100))

# %%
lr = 1e-5

change = np.Inf

i = 0

old_val_loss = 1e-15

while change >= 1e-15 and i<10000:
    grads = gradients(datas_train, labels_train,(w,b))
    w-=(grads[0] * lr)
    b-=(grads[1] * lr)

    val_loss = model_loss(datas_val,labels_val,(w,b))

    pred_labels_val = classify(datas_val, (w,b))
    score = performance(pred_labels_val, labels_val)

    change = np.abs((val_loss - old_val_loss)/old_val_loss)

    i += 1
    old_val_loss = val_loss

    if i% 50 == 0:
        print("{}...".format(i),end="")
print("")
print("")
print("Upon optimization stopped:")
print("    Iterations:", i)
print("    Validation loss:", val_loss)
print("    Validation precision:", score[0])
print("    Validation recall:", score[1])
print("    Validation F-score:", score[2])
print("    Validation Accuracy:", score[3])
print("    Change in validation loss:", change)


