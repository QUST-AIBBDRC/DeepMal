from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils


model = Sequential()
model.add(GRU(8,return_sequences=True)) 
model.add(Dropout(0.5))
model.add(GRU(4,return_sequences=True)) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(4, activation = 'relu',name="Dense_256"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax',name="Dense_2"))
model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics =['accuracy'])#rmsprop



data_=pd.read_csv(r'ALL_E_train.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X1=shu
y=label

X=np.reshape(X1,(-1,1,n1))

sepscores = []

ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

skf= StratifiedKFold(n_splits=10)

for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])#generate the resonable results
    cv_clf = model
    hist=cv_clf.fit(X[train], 
                    y_train,
                    epochs=19)
    
    y_score=cv_clf.predict(X[test])#the output of  probability
    y_class= utils.categorical_probas_to_classes(y_score)
    
    
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    yscore=np.vstack((yscore,y_score))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
   # fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    fpr, tpr, _ = roc_curve(y_test[:,1], y_score[:,1])
    roc_auc = auc(fpr, tpr)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    
scores=np.array(sepscores)
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores

row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_GRU_E.csv')

ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_GRU_E.csv')


data_csv = pd.DataFrame(data=result)
data_csv.to_csv('GRU_E_train.csv')

