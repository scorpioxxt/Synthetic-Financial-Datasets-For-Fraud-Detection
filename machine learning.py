#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import skew, boxcox
import os
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance,plot_tree,to_graphviz
from sklearn import metrics   #Additional scklearn functions


df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
print(df.info())
print(df.describe())
print(df.head())
print(len(df.duplicated(subset=['nameDest'],keep=False)))
#只保留TRANSFER 和 CASHOUT 这两个存在欺诈的交易类型
use_data=df[(df['type']=='TRANSFER')|(df['type']=='CASH_OUT')]
print(use_data.info())
#去掉对isFraud几乎没有影响的变量
use_data = use_data.drop(columns=['nameOrig','nameDest','isFlaggedFraud'],inplace=False)
print(use_data.info())
use_data = use_data.reset_index(drop=True)
#对type类型进行编码
type_encoder = preprocessing.LabelEncoder()
use_data_type = type_encoder.fit_transform(use_data['type'].values)
use_data['typecategory']=use_data_type
print(use_data.info())
#从相关性分析图中可以看出oldbalance和newbalance、amount和newbalance具有有一定相关性，故而可以用它们相减来获得新的特征
use_data['errorBalanceOrig'] = use_data.newbalanceOrig + use_data.amount - use_data.oldbalanceOrg
use_data['errorBalanceDest'] = use_data.oldbalanceDest + use_data.amount - use_data.newbalanceDest
X=use_data
Y=use_data['isFraud']
del X['isFraud']
del X['type']

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2,random_state = 5)
print(testX)
trainX = trainX.values
testX =testX.values

#用XGB进行学习
param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(max_depth=5,min_child_weight=1,scale_pos_weight=weights),
 param_grid = param_test1, scoring='average_precision',n_jobs=4,iid=False, cv=5)
gsearch1.fit(trainX, trainY)
print(gsearch1.best_params_, gsearch1.best_score_)
#得到效果最好的max_depth = 3, min_child_weight=3
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
clf = XGBClassifier(max_depth = 3, min_child_weight=3,scale_pos_weight = weights,n_jobs = -1)
probabilities = clf.fit(trainX, trainY).predict_proba(testX)
print(probabilities)
print('AUPRC = {}'.format(average_precision_score(testY,probabilities[:, 1])))
#绘制重要性排序图
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)
colours = plt.cm.Set1(np.linspace(0, 1, 9))
ax = plot_importance(clf, height=1, color=colours,grid=False,show_values=False, importance_type='cover', ax=ax)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.set_xlabel('importance score', size=16)
ax.set_ylabel('features', size=16)
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_title('Ordering of features by importance to the model learnt', size=20)
plt.show()
#绘制学习曲线
trainSizes, trainScores, crossValScores = learning_curve(XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = -1), trainX,\
                                         trainY, scoring = 'average_precision')
trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores, axis=1)
crossValScoresMean = np.mean(crossValScores, axis=1)
crossValScoresStd = np.std(crossValScores, axis=1)
colours = plt.cm.tab10(np.linspace(0, 1, 9))
fig = plt.figure(figsize = (14, 9))
plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
    trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
    crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', \
         color = colours[0])
plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val', \
         color = colours[1])
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15), \
               loc=2, borderaxespad=0, fontsize = 16);
plt.xlabel('training set size', size = 16);
plt.ylabel('AUPRC', size = 16)
plt.title('Learning curves indicate slightly underfit model', size = 20);
plt.show()