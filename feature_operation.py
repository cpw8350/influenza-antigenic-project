# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:21:59 2022
@author: cpww
"""

import pandas as pd
import numpy as np
#0
d=pd.read_excel('sequence.xlsx')

d['Sa1']=d['data1'].str[123:125]+d['data1'].str[152:157]+d['data1'].str[158:164]
d['Sb1']=d['data1'].str[183:195]
d['Ca1']=d['data1'].str[136:142]+d['data1'].str[165:170]+d['data1'].str[202:205]+d['data1'].str[220:223]+d['data1'].str[234:238]
d['Cb1']=d['data1'].str[69:75]
d['Sa2']=d['data2'].str[123:125]+d['data2'].str[152:157]+d['data2'].str[158:164]
d['Sb2']=d['data2'].str[183:195]
d['Ca2']=d['data2'].str[136:142]+d['data2'].str[165:170]+d['data2'].str[202:205]+d['data2'].str[220:223]+d['data2'].str[234:238]
d['Cb2']=d['data2'].str[69:75]
d['cross1']=d['data1'].str[177:181]
d['cross2']=d['data2'].str[177:181]
#######
#HA
sa=list()
sb=list()
ca=list()
cb=list()
cross=list()
for i in range(len(d)):
    sa.append(len(d.iloc[i]['Sa1'])-sum(a==b for a,b in zip(d.iloc[i]['Sa1'],d.iloc[i]['Sa2'])))
    sb.append(len(d.iloc[i]['Sb1'])-sum(a==b for a,b in zip(d.iloc[i]['Sb1'],d.iloc[i]['Sb2'])))
    ca.append(len(d.iloc[i]['Ca1'])-sum(a==b for a,b in zip(d.iloc[i]['Ca1'],d.iloc[i]['Ca2'])))
    cb.append(len(d.iloc[i]['Cb1'])-sum(a==b for a,b in zip(d.iloc[i]['Cb1'],d.iloc[i]['Cb2'])))
    cross.append(len(d.iloc[i]['cross1'])-sum(a==b for a,b in zip(d.iloc[i]['cross1'],d.iloc[i]['cross2'])))
    
    
#pp
data = pd.concat([d['seq1'],d['seq2'],d['data1'],d['data2']],axis=1)
#hydro
Hydrophobicity={'A':'-0.21','L':'-4.68','R':'2.11','K':'3.88','N':'0.96','M':'-3.66','D':'1.36',
    'F':'-4.65','C':'-6.04','P':'0.75','Q':'1.52','S':'1.74','E':'2.30','T':'0.78','G':'0.00',
    'W':'-3.32','H':'-1.23','Y':'-1.01','I':'-4.81','V':'-3.50','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#volume
volumn={'A':'31','L':'111','R':'124','K':'119','N':'56','M':'105','D':'54','F':'132','C':'55','P':'32.5',
    'Q':'85','S':'32','E':'83','T':'61','G':'3','W':'170','H':'96',
    'Y':'136','I':'111','V':'84','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#charge
charge={'A':'6','L':'5.98','R':'10.76','K':'9.74','N':'5.41','M':'5.74','D':'2.77','F':'5.48',
    'C':'5.05','P':'6.3','Q':'5.65','S':'5.68','E':'3.22','T':'5.66','G':'5.97','W':'5.89',
    'H':'7.59','Y':'5.66','I':'6.02','V':'5.96','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#polarity
polarity={'A':'0.046','L':'0.186','R':'0.291','K':'0.219','N':'0.134','M':'0.221','D':'0.105','F':'0.29',
    'C':'0.128','P':'0.131','Q':'0.18','S':'0.062','E':'0.151','T':'0.108','G':'0',
    'W':'0.409','H':'0.23','Y':'0.298','I':'0.186','V':'0.14','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#Average accessible surface area
AASA={'A':'27.8','L':'27.6','R':'94.7','K':'103','N':'60.1','M':'33.5','D':'60.6',
    'F':'25.5','C':'15.5','P':'51.5','Q':'68.7','S':'42','E':'68.2','T':'45','G':'24.5',
    'W':'34.7','H':'50.7','Y':'55.2','I':'22.8','V':'23.7','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#
import heapq
import numpy as np
hydro=list()
for i in range(len(d)):
    Hydro1=[]
    Hydro2=[]
    for j in range(len(d['data1'][1])):
        Hydro1.append(float(Hydrophobicity[d['data1'][i][j]]))
        Hydro2.append(float(Hydrophobicity[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(Hydro2,Hydro1)))
    hydro.append(np.mean(heapq.nlargest(3, m)))
volume=list()
for i in range(len(d)):
    vo1=[]
    vo2=[]
    for j in range(len(d['data1'][1])):
        vo1.append(float(volumn[d['data1'][i][j]]))
        vo2.append(float(volumn[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(vo2,vo1)))
    volume.append(np.mean(heapq.nlargest(3, m)))
cha=list()
for i in range(len(d)):
    cha1=[]
    cha2=[]
    for j in range(len(d['data1'][1])):
        cha1.append(float(charge[d['data1'][i][j]]))
        cha2.append(float(charge[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(cha2,cha1)))
    cha.append(np.mean(heapq.nlargest(3, m)))
po=list()
for i in range(len(d)):
    polarity1=[]
    polarity2=[]
    for j in range(len(d['data1'][1])):
        polarity1.append(float(polarity[d['data1'][i][j]]))
        polarity2.append(float(polarity[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(polarity2,polarity1)))
    po.append(np.mean(heapq.nlargest(3, m)))    
aasa=list()
for i in range(len(d)):
    AASA1=[]
    AASA2=[]
    for j in range(len(d['data1'][1])):
        AASA1.append(float(AASA[d['data1'][i][j]]))
        AASA2.append(float(AASA[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(AASA2,AASA1)))
    aasa.append(np.mean(heapq.nlargest(3, m))) 

#RBS
import pandas as pd
import numpy as np
import heapq
#
axis=pd.read_excel('axis.xlsx')
rbs1=pd.read_excel('130_loop.xlsx')
rbs2=pd.read_excel('190_helix.xlsx')
rbs3=pd.read_excel('220_loop.xlsx')
y=list()
for j in range(len(d)):
    r=[]
    for i in range(0,322):
        if d['data1'][j][i] != d['data2'][j][i]:  
            a=np.array(axis.iloc[i][1:4])
            d1=[]
            d2=[]
            d3=[]
            for s in range(len(rbs1)):
                b1=np.array(rbs1.iloc[s][6:9])
                d1.append(np.linalg.norm(a-b1))
                d1=heapq.nsmallest(1,d1)
            for s in range(len(rbs2)):
                b2=np.array(rbs2.iloc[s][6:9])
                d2.append(np.linalg.norm(a-b2))
                d2=heapq.nsmallest(1,d2)
                
            for s in range(len(rbs3)):
                b3=np.array(rbs3.iloc[s][6:9])
                d3.append(np.linalg.norm(a-b3))
                d3=heapq.nsmallest(1,d3)                    
                #r.append(sum(d1,d2,d3))                
            r.append([i + j +k for i, j, k in zip(d1,d2,d3)])
    y.append(np.mean(heapq.nsmallest(3,r)))    
x1=pd.DataFrame({'sa':sa,'sb':sb,'ca':ca,'cb':cb,'cross':cross,'hydro':hydro,'volume':volume,'char':cha,'polar':po,'aasa':aasa,'RBS':y})   
x1=x1.fillna(0)

              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
             









d['Sa1']=d['data1'].str[123:125]+d['data1'].str[152:157]+d['data1'].str[158:164]
d['Sb1']=d['data1'].str[183:195]
d['Ca1']=d['data1'].str[136:142]+d['data1'].str[165:170]+d['data1'].str[202:205]+d['data1'].str[220:223]+d['data1'].str[234:238]
d['Cb1']=d['data1'].str[69:75]
d['Sa2']=d['data2'].str[123:125]+d['data2'].str[152:157]+d['data2'].str[158:164]
d['Sb2']=d['data2'].str[183:195]
d['Ca2']=d['data2'].str[136:142]+d['data2'].str[165:170]+d['data2'].str[202:205]+d['data2'].str[220:223]+d['data2'].str[234:238]
d['Cb2']=d['data2'].str[69:75]
d['cross1']=d['data1'].str[177:181]
d['cross2']=d['data2'].str[177:181]
#######
#HA
sa=list()
sb=list()
ca=list()
cb=list()
cross=list()
for i in range(len(d)):
    sa.append(len(d.iloc[i]['Sa1'])-sum(a==b for a,b in zip(d.iloc[i]['Sa1'],d.iloc[i]['Sa2'])))
    sb.append(len(d.iloc[i]['Sb1'])-sum(a==b for a,b in zip(d.iloc[i]['Sb1'],d.iloc[i]['Sb2'])))
    ca.append(len(d.iloc[i]['Ca1'])-sum(a==b for a,b in zip(d.iloc[i]['Ca1'],d.iloc[i]['Ca2'])))
    cb.append(len(d.iloc[i]['Cb1'])-sum(a==b for a,b in zip(d.iloc[i]['Cb1'],d.iloc[i]['Cb2'])))
    cross.append(len(d.iloc[i]['cross1'])-sum(a==b for a,b in zip(d.iloc[i]['cross1'],d.iloc[i]['cross2'])))
    

    
  
#理化性质
data = pd.concat([d['seq1'],d['seq2'],d['data1'],d['data2']],axis=1)
#疏水性
Hydrophobicity={'A':'-0.21','L':'-4.68','R':'2.11','K':'3.88','N':'0.96','M':'-3.66','D':'1.36',
    'F':'-4.65','C':'-6.04','P':'0.75','Q':'1.52','S':'1.74','E':'2.30','T':'0.78','G':'0.00',
    'W':'-3.32','H':'-1.23','Y':'-1.01','I':'-4.81','V':'-3.50','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#volume
volumn={'A':'31','L':'111','R':'124','K':'119','N':'56','M':'105','D':'54','F':'132','C':'55','P':'32.5',
    'Q':'85','S':'32','E':'83','T':'61','G':'3','W':'170','H':'96',
    'Y':'136','I':'111','V':'84','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#charge
charge={'A':'6','L':'5.98','R':'10.76','K':'9.74','N':'5.41','M':'5.74','D':'2.77','F':'5.48',
    'C':'5.05','P':'6.3','Q':'5.65','S':'5.68','E':'3.22','T':'5.66','G':'5.97','W':'5.89',
    'H':'7.59','Y':'5.66','I':'6.02','V':'5.96','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#polarity
polarity={'A':'6','L':'5.98','R':'10.76','K':'9.74','N':'5.41','M':'5.74','D':'2.77','F':'5.48',
    'C':'5.05','P':'6.3','Q':'5.65','S':'5.68','E':'3.22','T':'5.66','G':'5.97',
    'W':'5.89','H':'7.59','Y':'5.66','I':'6.02','V':'5.96','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#Average accessible surface area
AASA={'A':'27.8','L':'27.6','R':'94.7','K':'103','N':'60.1','M':'33.5','D':'60.6',
    'F':'25.5','C':'15.5','P':'51.5','Q':'68.7','S':'42','E':'68.2','T':'45','G':'24.5',
    'W':'34.7','H':'50.7','Y':'55.2','I':'22.8','V':'23.7','-':'0','X':'0','B':'0','J':'0','Z':'0',}
#全部抗原表位的疏水性值
import heapq
import numpy as np
hydro=list()
for i in range(len(d)):
    Hydro1=[]
    Hydro2=[]
    for j in range(len(d['data1'][1])):
        Hydro1.append(float(Hydrophobicity[d['data1'][i][j]]))
        Hydro2.append(float(Hydrophobicity[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(Hydro2,Hydro1)))
    hydro.append(np.mean(heapq.nlargest(3, m)))
volume=list()
for i in range(len(d)):
    vo1=[]
    vo2=[]
    for j in range(len(d['data1'][1])):
        vo1.append(float(volumn[d['data1'][i][j]]))
        vo2.append(float(volumn[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(vo2,vo1)))
    volume.append(np.mean(heapq.nlargest(3, m)))
cha=list()
for i in range(len(d)):
    cha1=[]
    cha2=[]
    for j in range(len(d['data1'][1])):
        cha1.append(float(charge[d['data1'][i][j]]))
        cha2.append(float(charge[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(cha2,cha1)))
    cha.append(np.mean(heapq.nlargest(3, m)))
po=list()
for i in range(len(d)):
    polarity1=[]
    polarity2=[]
    for j in range(len(d['data1'][1])):
        polarity1.append(float(polarity[d['data1'][i][j]]))
        polarity2.append(float(polarity[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(polarity2,polarity1)))
    po.append(np.mean(heapq.nlargest(3, m)))    
aasa=list()
for i in range(len(d)):
    AASA1=[]
    AASA2=[]
    for j in range(len(d['data1'][1])):
        AASA1.append(float(AASA[d['data1'][i][j]]))
        AASA2.append(float(AASA[d['data2'][i][j]]))   
    m=list(map(lambda x: abs(x[0]-x[1]), zip(AASA2,AASA1)))
    aasa.append(np.mean(heapq.nlargest(3, m))) 

#RBS
import pandas as pd
import numpy as np
import heapq
#数据导入
axis=pd.read_excel('坐标信息.xlsx')
rbs1=pd.read_excel('130_loop.xlsx')
rbs2=pd.read_excel('190_helix.xlsx')
rbs3=pd.read_excel('220_loop.xlsx')
y=list()
for j in range(len(d)):
    r=[]
    for i in range(0,322):
        if d['data1'][j][i] != d['data2'][j][i]:  
            a=np.array(axis.iloc[i][1:4])
            d1=[]
            d2=[]
            d3=[]
            for s in range(len(rbs1)):
                b1=np.array(rbs1.iloc[s][6:9])
                d1.append(np.linalg.norm(a-b1))
                d1=heapq.nsmallest(1,d1)
            for s in range(len(rbs2)):
                b2=np.array(rbs2.iloc[s][6:9])
                d2.append(np.linalg.norm(a-b2))
                d2=heapq.nsmallest(1,d2)
                
            for s in range(len(rbs3)):
                b3=np.array(rbs3.iloc[s][6:9])
                d3.append(np.linalg.norm(a-b3))
                d3=heapq.nsmallest(1,d3)                    
                #r.append(sum(d1,d2,d3))                
            r.append([i + j +k for i, j, k in zip(d1,d2,d3)])
    y.append(np.mean(heapq.nsmallest(3,r)))    
x1=pd.DataFrame({'sa':sa,'sb':sb,'ca':ca,'cb':cb,'hydro':hydro,'volume':volume,'char':cha,'polar':po,'aasa':aasa,'RBS':y})   
x1=x1.fillna(0)

'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #训练测试集拆分
from sklearn.linear_model import LogisticRegression  #逻辑回归模型
import matplotlib.pyplot as plt #画图函数
#from sklearn.externals import joblib #保存加载模型函数joblib
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc



x=data.iloc[:,[3,4,5,6,7,8,9,10,11]]
y=data['sort']
#拆分训练集和测试集（7:3）
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=666, train_size = 0.7)


from xgboost import XGBClassifier
my_model = XGBClassifier()
my_model.fit(x_train, y_train)
# 计算指标参数
y_pred=my_model.predict(x_test)
#计算混淆矩阵
from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test, my_model.predict(x_test))
print("混淆矩阵：")
print(C)
#metrics计算
#准确度
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
#精确率
from sklearn.metrics import precision_score
print(precision_score(y_test,y_pred))
#recall
from sklearn.metrics import recall_score
print(recall_score(y_test,y_pred))
#F1
from sklearn.metrics import f1_score
print(f1_score(y_test,y_pred))
'''




