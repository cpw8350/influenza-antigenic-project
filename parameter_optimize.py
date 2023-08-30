# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 17:21:28 2022

@author: cpww
"""
#data
import pandas as pd
import numpy as np

new=pd.read_csv(r'D:\Project1\egghi_fea.csv')
new=new.drop_duplicates(subset=['seq1','seq2'],keep='first')
# In[]
x=new.iloc[:, :-3]
y=new['sort']
new['sort'] = new['sort'].astype('int')
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=666, train_size = 0.7)

#oversample
from imblearn. over_sampling import SMOTE
from collections import Counter
smote = SMOTE(random_state=0)  
X_smotesampled, y_smotesampled = smote.fit_resample(x, y)  

#training set:test set（7:3）
X_train, X_test, Y_train, Y_test = train_test_split(X_smotesampled, y_smotesampled, random_state=0, train_size = 0.7)
# In[]
#2009-2022 data
d=pd.read_csv(r'D:\Project1\crick\H1_21-22\feature09-22.csv')
d=d.drop_duplicates(subset=['seq1','seq2'],keep=False)

new=pd.read_csv(r'D:\Project1\crick\H1_21-22\hi21-22_feature.csv')
new=new.drop_duplicates(subset=['seq1','seq2'],keep=False)

n=pd.concat([d,new],axis=0)
n=n.drop_duplicates(subset=['seq1','seq2'],keep=False)
#n=n.dropna(axis=0)
n.to_csv('hi_feature09-22.csv')
x=n.iloc[:, :-4]
y=n['sort']
n['sort'] = n['sort'].astype('int')
from sklearn.model_selection import train_test_split #训练测试集拆分
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=666, train_size = 0.7)

#oversample
from imblearn. over_sampling import SMOTE
from collections import Counter
smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
X_smotesampled, y_smotesampled = smote.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
#training set:test set（7:3）
X_train, X_test, Y_train, Y_test = train_test_split(X_smotesampled, y_smotesampled, random_state=0, train_size = 0.7)

# In[]
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics   
from sklearn.model_selection import GridSearchCV

cv_params = {'n_estimators': [100,200,300,400,500]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_
#print('res_each_iteration:{0}'.format(evalute_result))
print('para_perform_best：{0}'.format(optimized_GBM.best_params_))
print('model_score:{0}'.format(optimized_GBM.best_score_))
#{'n_estimators': 6}
# In[]
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_
#print('res_each_iteration:{0}'.format(evalute_result))
print('para_perform_best：{0}'.format(optimized_GBM.best_params_))
print('model_score:{0}'.format(optimized_GBM.best_score_))
#{'max_depth': 10, 'min_child_weight': 1}

# In[]
cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 6, 'max_depth': 10, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_
#print('res_each_iteration:{0}'.format(evalute_result))
print('para_perform_best：{0}'.format(optimized_GBM.best_params_))
print('model_score:{0}'.format(optimized_GBM.best_score_))
#{'gamma': 0.1}
# In[]
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
other_params = {'learning_rate': 0.1, 'n_estimators':6, 'max_depth': 10, 'min_child_weight':1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_
#print('res_each_iteration:{0}'.format(evalute_result))
print('para_perform_best：{0}'.format(optimized_GBM.best_params_))
print('model_score:{0}'.format(optimized_GBM.best_score_))
#{'colsample_bytree': 0.8, 'subsample': 0.9}

# In[]
cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
other_params = {'learning_rate': 0.1, 'n_estimators': 6, 'max_depth': 10, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_
#print('res_each_iteration:{0}'.format(evalute_result))
print('para_perform_best：{0}'.format(optimized_GBM.best_params_))
print('model_score:{0}'.format(optimized_GBM.best_score_))
#{'reg_alpha': 0.05, 'reg_lambda': 0.05}

# In[]
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 6, 'max_depth': 10, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.9, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0.05, 'reg_lambda': 0.1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_
#print('res_each_iteration:{0}'.format(evalute_result))
print('para_perform_best：{0}'.format(optimized_GBM.best_params_))
print('model_score:{0}'.format(optimized_GBM.best_score_))
#{'learning_rate': 0.2}

# In[]
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
xgb1 = XGBClassifier(
     learning_rate=0.2, 
     n_estimators= 500,
     max_depth=10,
     min_child_weight=2,
     seed=0,
     subsample=0.9, 
     colsample_bytree=0.7, 
     gamma=0.1, 
     reg_alpha=0.1, 
     reg_lambda=0.05)
xgb1.fit(X_train,Y_train)
from sklearn import model_selection, metrics   
ypred1=xgb1.predict(x_test)
print(metrics.accuracy_score(y_test, ypred1))
# In[]
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
xgb1 = XGBClassifier(
     learning_rate=0.2,

     max_depth=10,
     min_child_weight=2,
     seed=0,
     subsample=0.9, 
     colsample_bytree=0.8, 
     gamma=0.1, 
     reg_alpha=0.05, 
     reg_lambda=0.05)
xgb1.fit(X_train,Y_train)
from sklearn import model_selection, metrics   
ypred1=xgb1.predict(x_test)
print(metrics.accuracy_score(y_test, ypred1))
# In[]  ten_fold
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#
scores = cross_val_score(xgb1, x, y,cv=10)
print("Cross validation score: {}".format(scores))
print("Average corss_val_score is: {}".format(scores.mean()))
# In[]
from sklearn.metrics import roc_auc_score
y_pred1=xgb1.predict(x_test)

from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test, xgb1.predict(x_test))
print("confusion_matrix：")
print(C)
#metrics
#acc
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred1))
#precision
from sklearn.metrics import precision_score
print(precision_score(y_test,y_pred1))
#recall
from sklearn.metrics import recall_score
print(recall_score(y_test,y_pred1))
#F1
from sklearn.metrics import f1_score
print(f1_score(y_test,y_pred1))
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = xgb1.predict_proba(x_test)[:,0]
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
Xgbc_auc=roc_auc_score(y_test,y_pred1) #Xgbc_auc值
print(Xgbc_auc)  

# In[] save
import pickle
pickle.dump(xgb1, open("xgb_0301.pkl", "wb"))

# In[] 
from sklearn.model_selection import KFold
def Split_Sets_10_Fold(total_fold, data):   
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index

total_fold = 10
[train_index, test_index] = Split_Sets_10_Fold(total_fold, new)
train_data = new[train_index, :, :, :]     
test_data = new[test_index, :,:,:]           	


# In[] feature importance
import shap
explainer = shap.TreeExplainer(xgb1)
shap_values = explainer.shap_values(x)
# Characteristic statistical values
shap.summary_plot(shap_values,x,max_display=12,plot_type="bar")

# In[]
x=['sa','sb','ca','cb','RF','hydro','hydro_index','TFE','size','RBS','netn','neto']
#y=[0.1138612,0.05182144,0.0546623,0.09405176,0.14200355,0.15348443,0.07030667,0.05290741,0.06390887,0.05532617,0.08405223,0.06361394]

y=list(xgb1.feature_importances_)
import matplotlib.pyplot as plt
plt.figure(dpi=300,figsize=(25,12))
plt.bar(x,y)
plt.tick_params(labelsize=23)
plt.grid(color='grey',
              linestyle='--',
              linewidth=1,
              alpha=0.3)
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42
plt.savefig('import.pdf')

# In[] 
feature_importance = pd.DataFrame()
feature_importance['feature'] = x.columns
feature_importance['importance'] = np.abs(shap_values).mean(0)
feature_importance.sort_values('importance', ascending=False)

'''
import pickle
xgb_model_loaded = pickle.load(open("xgb_0301.pkl", "rb"))
'''