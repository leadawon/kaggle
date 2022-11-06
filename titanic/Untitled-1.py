# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
datas = pd.read_csv("titanic/train.csv")
print(type(datas))
datas.head()
datas.info()
y_col = "Survived"
x_cols = datas.columns.drop(["PassengerId","Survived","Name","Ticket","Fare","Cabin","Embarked"])
print(x_cols)
#!pip install autograd
#!pip install sklearn
from autograd import numpy
from autograd import grad

# %%
for i in range(len(datas["Sex"])):
    if datas["Sex"][i] == "male":
        datas["Sex"][i] = 0
    else:
        datas["Sex"][i] = 1
mean_of_age=datas['Age'].mean()
isnalst = datas["Age"].isna()
for i in range(len(isnalst)):
    if isnalst[i]:
        datas["Age"][i] = mean_of_age

print(datas["Age"])
print(datas.head())

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(datas[x_cols])
X_scaled = numpy.hstack((numpy.ones((X_scaled.shape[0],1)),X_scaled))

y = datas[y_col].values

print(f"x.s={X_scaled.shape} y.s={y.shape}")


# %%
def logistic(z):
    return 1/(1+numpy.exp(-z))
def logistic_model(params,x):
    z = numpy.dot(X_scaled,params)
    y = logistic(z)
    return y
def log_loss(params,model,x,y,_lambda=1.0):
    y_pred = model(params,x)
    print(y_pred)
    return -numpy.mean(y * numpy.log(y_pred) + (1-y) * numpy.log(1 - y_pred)) \
#+ _lambda * numpy.sum(params[1:]**2)

# %%
gradient = grad(log_loss)

# %%
type(gradient)

# %%
max_iter = 300
alpha = 0.01
params = numpy.zeros(X_scaled.shape[1])
descent = numpy.ones(X_scaled.shape[1])
i = 0
print(params)
while (i<max_iter):
    descent = gradient(params,logistic_model,X_scaled,y)
    params = params - descent * alpha
    i += 1
    if i%10 == 0:
        print(f"--{params}--")


