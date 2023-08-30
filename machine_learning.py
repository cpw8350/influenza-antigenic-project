# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:58:04 2022

@author: cpww
"""

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
from sklearn.metrics import roc_auc_score

new=pd.read_csv('D:/练习用数据/egghi_fea.csv')
x=new.iloc[:, :-3]
y=new['sort']
new['sort'] = new['sort'].astype('int')
from sklearn.model_selection import train_test_split #训练测试集拆分
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=666, train_size = 0.7)

#过采样
from imblearn. over_sampling import SMOTE
from collections import Counter
smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
X_smotesampled, y_smotesampled = smote.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
#拆分训练集和测试集（7:3）
X_train, X_test, Y_train, Y_test = train_test_split(X_smotesampled, y_smotesampled, random_state=0, train_size = 0.7)


# In[]
#lr
lr= LogisticRegression()
lr.fit(X_train, Y_train)
#预测结果，并评测
y_pred = lr.predict(x_test)  
y_true = y_test              
target_names = ['class 0', 'class 1']
#print(classification_report(y_true, y_pred, target_names=target_names)) #可以参考sklearn官网API
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred)) 
#merics
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
#roc计算
y_score=lr.fit(x_train, y_train).decision_function(x_test)
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
lr_auc=roc_auc_score(y_test,y_pred) #Xgbc_auc值
print(lr_auc)  
# In[]
import pickle
pickle.dump(lr, open("linear_reg.pkl", "wb"))

# In[]
#SVM

from sklearn.svm import SVC
svm = SVC(kernel='linear', probability=True,random_state=0)
svm.fit(X_train,Y_train)
y_pred = svm.predict(x_test)  #预测出来的值计做y_pred
y_true = y_test               
target_names = ['class 0', 'class 1']
#print(classification_report(y_true, y_pred, target_names=target_names)) #可以参考sklearn官网API
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred)) #行真实，列预测
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
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = svm.fit(x_train, y_train).decision_function(x_test)
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
svm_auc=roc_auc_score(y_test,y_pred) #Xgbc_auc值
print(svm_auc) 
# In[]
import pickle
pickle.dump(svm, open("svm.pkl", "wb"))

# In[]
#lgbm

import lightgbm as lgb
clf=lgb.LGBMClassifier()
clf.fit(X_train,Y_train)
y_pred=clf.predict(x_test)
#计算混淆矩阵
from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test, clf.predict(x_test))
print("混淆矩阵：")
print(C)

#metrics
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
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = clf.predict_proba(x_test)[:,0]
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
lgbm_auc=roc_auc_score(y_test,y_pred) #Xgbc_auc值
print(lgbm_auc)  

#判断过拟合与否
#train_set_score
clf.score(x_train,y_train)
#test_set_score
clf.score(x_test,y_pred)

#分类报告
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
# In[]
import pickle
pickle.dump(clf, open("LGBM.pkl", "wb"))

# In[]
#rf
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    criterion='entropy',
    n_estimators=3, 
    max_depth=None, # 定义树的深度, 可以用来防止过拟合
    min_samples_split=10, # 定义至少多少个样本的情况下才继续分
    #min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
    )
# 模型训练
rf.fit(X_train, Y_train)
# 计算指标参数
rf.predict(x_test)
rf_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
print ("随机森林 AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(x_test)))
y_test[y_test==1]
#计算混淆矩阵
from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test, rf.predict(x_test))
print("混淆矩阵：")
print(C)
#metrics计算
y_pred=rf.predict(x_test)
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
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = rf.predict_proba(x_test)[:,0]
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
rf_auc=roc_auc_score(y_test,y_pred) #Xgbc_auc值
print(rf_auc) 
#from sklearn.linear_model import LogisticRegression
#创建一个模拟数据集
scores = cross_val_score(rf, x, y,cv=10,scoring = "roc_auc")
print("Cross validation score: {}".format(scores))
print("Average corss_val_score is: {}".format(scores.mean()))
# In[]
import pickle
pickle.dump(rf, open("RF.pkl", "wb"))

# In[]
from xgboost.sklearn import XGBClassifier
#xgb
xgb1 = XGBClassifier(
     learning_rate=0.2,  
     max_depth=10,
     min_child_weight=2,
     seed=0,
     subsample=0.9, 
     colsample_bytree=0.7, 
     gamma=0.1, 
     reg_alpha=0.1, 
     reg_lambda=0.05)
xgb1.fit(X_train,Y_train)
y_pred=xgb1.predict(x_test)

#计算混淆矩阵
from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test, xgb1.predict(x_test))
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
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = xgb1.predict_proba(x_test)[:,0]
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
Xgbc_auc=roc_auc_score(y_test,y_pred) #Xgbc_auc值
print(Xgbc_auc)  



# In[]  xgboost十折交叉验证
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#创建一个模拟数据集
scores = cross_val_score(xgb1, x, y,cv=10)
print("Cross validation score: {}".format(scores))
print("Average corss_val_score is: {}".format(scores.mean()))

# In[]
#导入包
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn import metrics
from sklearn.model_selection import train_test_split

#函数编写
def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
        """
        将多个机器模型的roc图输出到一张图上       
        Args:
            names: list, 多个模型的名称
            sampling_methods: list, 多个模型的实例化对象
            save: 选择是否将结果保存（默认为png格式）
            dpin控制图片的信息量（其实可以理解为清晰度           
        Returns:
            返回图片对象plt
        """
        plt.figure(figsize=(20,25), dpi=400)
        # figsize控制图片大小
        for (name, method, colorname) in zip(names, sampling_methods, colors):
            y_test_preds = method.predict(x_test)
            y_test_predprob = method.predict_proba(x_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
            plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)),color = colorname)
            plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
            plt.axis('square')
            plt.xlim([0, 1])
            plt.tick_params(labelsize=30)
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate',fontsize=50,labelpad=30)
            plt.ylabel('True Positive Rate',fontsize=50,labelpad=30)
            plt.legend(loc='lower right',fontsize=37)
            '''
        if save:
            plt.savefig('multi_models_roc.png')
            '''
        return plt
# In[]
names = ['Logistic Regression',
         'SVM',
         'LGBM',
         'Random Forest',
         'XGBoost']

sampling_methods = [lr,
                    svm,
                    rf,
                    clf,
                    xgb1]

colors = ['crimson',
          'orange',          
          'mediumseagreen',
          'steelblue', 
          'mediumpurple']

#ROC curves
train_roc_graph = multi_models_roc(names, sampling_methods, colors, x_train, y_train, save = True)
#train_roc_graph.savefig('ROC_Train_all.pdf')
# In[]
#ROC curves
train_roc_graph = multi_models_roc(names, sampling_methods, colors, x_test, y_test, save = True)
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42
train_roc_graph.savefig('ROC_TEST_all.pdf')



# In[]

#导入包
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn import metrics
from sklearn.model_selection import train_test_split

#函数编写
def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
        """
        将多个机器模型的roc图输出到一张图上       
        Args:
            names: list, 多个模型的名称
            sampling_methods: list, 多个模型的实例化对象
            save: 选择是否将结果保存（默认为png格式）
            dpin控制图片的信息量（其实可以理解为清晰度           
        Returns:
            返回图片对象plt
        """
        plt.figure(figsize=(20,25), dpi=400)
        # figsize控制图片大小
        for (name, method, colorname) in zip(names, sampling_methods, colors):
            y_test_preds = method.predict(X_test)
            y_test_predprob = method.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
            plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)),color = colorname)
            plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
            plt.axis('square')
            plt.xlim([0, 1])
            plt.tick_params(labelsize=30)
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate',fontsize=50,labelpad=30)
            plt.ylabel('True Positive Rate',fontsize=50,labelpad=30)
            plt.legend(loc='lower right',fontsize=37)
            '''
        if save:
            plt.savefig('multi_models_roc.png')
            '''
        return plt
# In[]
names = ['Logistic Regression',
         'SVM',
         'LGBM',
         'Random Forest',
         'XGBoost']

sampling_methods = [lr,
                    svm,
                    rf,
                    clf,
                    xgb1]

colors = ['crimson',
          'orange',          
          'mediumseagreen',
          'steelblue', 
          'mediumpurple']

#ROC curves
train_roc_graph = multi_models_roc(names, sampling_methods, colors, X_train, Y_train, save = True)
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42    
train_roc_graph.savefig('ROC_trainall.pdf')