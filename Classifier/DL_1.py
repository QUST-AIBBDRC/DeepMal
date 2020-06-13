import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from keras.layers import Conv1D, MaxPooling1D,Dropout
from keras.models import Model




model = Sequential()
model.add(Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation= 'relu'))
model.add(MaxPooling1D(pool_size = 4))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation = 'relu',name="Dense_128"))
model.add(Dense(64, activation = 'relu',name="Dense_64"))
model.add(Dense(32, activation = 'relu',name="Dense_32"))
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
X=shu
y=label

sepscores = []

ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

X = X.reshape(-1, 53, 27)

skf= StratifiedKFold(n_splits=10)

for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])#generate the resonable results
    cv_clf = model
    hist=cv_clf.fit(X[train], 
                    y_train,
                    epochs=19)
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X[test])#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('GTB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    #hist=[]
    #cv_clf=[]
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_DL_E_train_19.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_DL_E_train_19.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='DL ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('cnn1_dnn_E_train_19.csv')

