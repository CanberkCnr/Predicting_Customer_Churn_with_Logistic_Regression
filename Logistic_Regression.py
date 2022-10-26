#Libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

# We will predict costumer churn for telecommunications. Our dataset is telecommunications dataset.

#You can download data via this link: 
#https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv
churn_df = pd.read_csv("ChurnData.csv")
#churn_df.head()

#We change the target data type to be an integer.
churn_df = churn_df[["tenure", "age", "address", "income", "ed", "employ", "equip", "callcard", "wireless", "churn"]]
churn_df["churn"] =churn_df["churn"].astype("int")
#churn_df.head()

churn_df.shape
#Output:(200, 10)

#Define X
X = np.asarray(churn_df[["tenure", "age", "address", "income", "ed", "employ", "equip"]])
X[0:5]

#Define Y
Y = np.asarray(churn_df["churn"])
Y[0:5]

#Normalize the Dataset
#from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#Train/Test Dataset
#Split Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=4)

#If we want to see Train and Test split, we can print shape.
#print ('Train set:', X_train.shape,  Y_train.shape)
#print ('Test set:', X_test.shape,  Y_test.shape)

#Modeling(Logistic Regression with Scikit-Learn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver="liblinear").fit(X_train,Y_train) # C = inverse of regularization strength , Numerical Optimizers can be "newton-cg", "lbfgs", "libliner", "sag", "saga"
LR

#Predict
yhat = LR.predict(X_test)
yhat

#Predict_proba is estimates for all classes. 
#First column is class 0, P(Y=0|X)
#Second column is class 1, P(Y=1|X)
yhat_prob = LR.predict_proba(X_test)
yhat_prob

#Evaluation

#Jaccard Index
from sklearn.metrics import jaccard_score

jaccard_score(Y_test,yhat,pos_label=0)

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes, normalize = False, title= "Confusion Matrix", cmap = plt.cm.Blues):

    if normalize:
        cm = cm.astype("float") / cm.sum(axis = 1)[:,np.newaxis]
        print("Normaliezed Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")
    print(cm)

    #PLOT informations
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(Y_test, yhat, labels=[1,0]))

#Compute Confusion Matrix
cnf_matrix = confusion_matrix(Y_test,yhat,labels = [1,0])
np.set_printoptions(precision =2)

#Plot Non-Normalized Confusion Matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ["churn = 1","churn = 0"],normalize = False, title = "Confusion Matrix")
print (classification_report(Y_test, yhat))
# Precision = TP/(TP+FP)
# Recall = TP/(TP+FN)
# F1 score: Its best value at 1 and worst value at 0.

#Log loss
from sklearn.metrics import log_loss
log_loss(Y_test, yhat_prob)

#Build Logistic Regression Model Again but this time use different solver and regularization values.
LR_2 = LogisticRegression(C=0.01, solver="newton-cg").fit(X_train,Y_train)
LR_2
#Prdict
yhat2 = LR_2.predict(X_test)
yhat2
#Predict_proba
yhat_prob_2 = LR_2.predict_proba(X_test)
yhat_prob_2
log_loss(Y_test, yhat_prob_2)