import numpy as np

t = open('label_M_test.txt','w')
#t = open('label_test.txt','w')
num=300
label1=np.ones((num,1))
label2=np.zeros((num,1))

for i in range(num):
    zifu='000000'
    zifuchuan=zifu+str(i+1)+' '+str(int(label1[i]))
    t.write(zifuchuan +'\n')
    zifu=[]
    da1=[]
    
for j in range(num):
    zifu='000000'
    zifuchuan=zifu+str(j+num+1)+' '+str(int(label2[j]))
    t.write(zifuchuan +'\n')
    zifu=[]
    da1=[]
t.close()
