from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv("second.kddcup.data_10_percent.corrected.csv")

data.columns=['0','1','2','3','4','5','6','7']


X_train, X_test, y_train, y_test =train_test_split(data.iloc[:,:7],data['7'].astype(int),test_size=0.8)

# # 寻找C和gamma的粗略范围
# CScale = [i for i in range(100,201,10)];
# gammaScale = [i/10 for i in range(1,11)];
# cv_scores = 0
# for i in CScale:
#     for j in gammaScale:
#         model = SVC(kernel = 'rbf', C = i,gamma=j)
#          # 交叉验证
#     # sklearn.cross_validation.cross_val_score(estimator, X, y=None, scoring=None,
#     #   cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’)
#
#         scores = model_selection.cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy')
#         if scores.mean()>cv_scores:
#             cv_scores = scores.mean()
#             savei = i
#             savej = j*100
#
# # # 找到更精确的C和gamma
# CScale = [i for i in range(savei-5,savei+5)];
# gammaScale = [i/100+0.01 for i in range(int(savej)-5,int(savej)+5)];
# cv_scores = 0
# for i in CScale:
#     for j in gammaScale:
#         model = SVC(kernel = 'rbf', C = i,gamma=j)
#         scores = model_selection.cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy')
#         if scores.mean()>cv_scores:
#             cv_scores = scores.mean()
#             savei = i
#             savej = j



model = SVC(kernel='rbf', C=600, gamma=0.4)
# scores = model_selection.cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy')
model.fit(X_train, y_train)#训练数据模型
pre = model.predict(X_test)
model.score(X_test,y_test)#测试数据模型
print(model.score(X_test,y_test))


#绘制混淆矩阵
cm = confusion_matrix(y_test, pre, labels=[-1, 1], sample_weight=None)#混淆矩阵
sns.set()
f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax) #画热力图
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()




