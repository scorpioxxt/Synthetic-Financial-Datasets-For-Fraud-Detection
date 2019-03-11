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
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance,plot_tree,to_graphviz


df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
print(df.info())
print(df.describe())
print(df.head())
print(len(df.duplicated(subset=['nameDest'],keep=False)))
#分析交易类型和欺诈的关系
df.groupby(['type','isFraud'])['isFraud'].count().plot.bar()
plt.show()
df.groupby(['type','isFlaggedFraud'])['isFlaggedFraud'].count().plot.bar()
plt.show()
df[['type','isFraud']].groupby(['type']).mean().plot.bar()
plt.show()
df[['type','isFlaggedFraud']].groupby(['type']).mean().plot.bar()
plt.show()
df[['isFraud','isFlaggedFraud']].groupby(['isFraud']).mean().plot.bar()
plt.show()
#只有TRANSFER类型会有交易被系统标注为欺诈
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
transfer_data = df[df['type'] == 'TRANSFER']
a = sns.boxplot(x='isFlaggedFraud', y='amount', data=transfer_data, ax=axs[0][0])
axs[0][0].set_yscale('log')   #查看的是转账金额与系统是否标注为欺诈 之间的关系，通过数据可视化发现被标注为欺诈的转账金额往往较高。
b = sns.boxplot(x='isFlaggedFraud', y='oldbalanceDest', data=transfer_data, ax=axs[0][1])  #目标账户原先的余额 系统是否标注为欺诈之间的关系 欺诈的原先账户余额往往较少
axs[0][1].set(ylim=(0, 0.5e8))  # ylim限制y轴的范围
c = sns.boxplot(x='isFlaggedFraud', y='oldbalanceOrg', data=transfer_data, ax=axs[1][0])  #向外转账的账户原先的余额 与系统是否标注为欺诈之间的关系
axs[1][0].set(ylim=(0, 3e7))    #箱图的结果基本符合主观常识
d = sns.regplot(x='oldbalanceOrg', y='amount', data=transfer_data[transfer_data['isFlaggedFraud'] ==1], ax=axs[1][1])#线性关系？原先账户的余额越多转出的就越多？
plt.show()


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

#探索变量间的相关关系
sns.heatmap(use_data.corr())
plt.show()
#把样本分为欺诈和非欺诈两种
Xfraud = use_data[use_data['isFraud']==0]
Xgenie = use_data[use_data['isFraud']==1]

#从相关性分析图中可以看出oldbalance和newbalance、amount和newbalance具有有一定相关性，故而可以用它们相减来获得新的特征
use_data['errorBalanceOrig'] = use_data.newbalanceOrig + use_data.amount - use_data.oldbalanceOrg
use_data['errorBalanceDest'] = use_data.oldbalanceDest + use_data.amount - use_data.newbalanceDest
X=use_data
Y=use_data['isFraud']
del X['isFraud']
#定义函数绘制散点图
limit = len(use_data)
def plotStrip(x, y, hue, figsize=(14, 9)):
    fig = plt.figure(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y, \
                           hue=hue, jitter=0.4, marker='.', \
                           size=4, palette=colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size=16)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), \
                   loc=2, borderaxespad=0, fontsize=16);
    return ax
#观察step和是否欺诈间的关系
ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 16)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent \
transactions over time', size = 20)
plt.show()
#观察amount和是否欺诈间的关系
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine \
and fraudulent transactions over amount', size = 18)
plt.show()
#观察errorBalanceDest和是否欺诈间的关系
ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('- errorBalanceDest', size = 16)
ax.set_title('Opposite polarity fingerprints over the error in \
destination account balances', size = 18)
plt.show()
#观察errorBalanceOrig和是否欺诈间的关系
ax = plotStrip(Y[:limit], - X.errorBalanceOrig[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('- errorBalanceOrig', size = 16)
ax.set_title('Opposite polarity fingerprints over the error in \
destination account balances', size = 18)
plt.show()
#观察errorBalanceDest、step、errorBalanceOrig间的相关关系
x = 'errorBalanceDest'
y = 'step'
z = 'errorBalanceOrig'
zOffset = 0.02
limit = len(X)
sns.reset_orig()  # prevent seaborn from over-riding mplot3d defaults
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.loc[Y == 0, x][:limit], X.loc[Y == 0, y][:limit], \
           -np.log10(X.loc[Y == 0, z][:limit] + zOffset), c='g', marker='.', \
           s=1, label='genuine')
ax.scatter(X.loc[Y == 1, x][:limit], X.loc[Y == 1, y][:limit], \
           -np.log10(X.loc[Y == 1, z][:limit] + zOffset), c='r', marker='.', \
           s=1, label='fraudulent')
ax.set_xlabel(x, size=16);
ax.set_ylabel(y + ' [hour]', size=16);
ax.set_zlabel('- log$_{10}$ (' + z + ')', size=16)
ax.set_title('Error-based features separate out genuine and fraudulent \
transactions', size = 20)
plt.axis('tight')
ax.grid(1)
noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='g', marker='.',
                          markersize = 10, label='genuine')
fraudMarker = mlines.Line2D([], [], linewidth = 0, color='r', marker='.',
                          markersize = 10, label='fraudulent')
plt.legend(handles = [noFraudMarker, fraudMarker], \
           bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16})
plt.show()






