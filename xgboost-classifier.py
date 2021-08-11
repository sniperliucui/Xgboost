# -*- coding: utf-8 -*-
"""
@author: away
@software: PyCharm
@file: xgboost_train.py
@time: 2021/8/10 11:44 下午
"""

from matplotlib import pyplot
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.datasets import make_hastie_10_2  # 随机生层2分类10维度样本
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')

# 使用鸢尾花做测试
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)  # test_size测试集合所占比例
model = XGBClassifier(

    silent=1,  # 设置成1则没有运行信息输出，设置为0，在运行升级时打印消息。
    learning_rate=0.3,  # 学习率
    min_child_weight=1,
    nthread=4,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 over_fitting。
    max_depth=6,  # 构建树的深度，越大越容易过拟合
    gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=1,  # 随机采样训练样本 训练实例的子采样比
    max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1,  # 生成树时进行的列采样
    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    n_estimators=1000,  # 树的个数
    seed=1000  # 随机种子

)

model.fit(X_train, y_train, eval_metric='auc')

y_true, y_pred = y_test, model.predict(X_test)

print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))

model.fit(X, y)
plot_importance(model)
pyplot.savefig('importance_feature.png')
# pyplot.show()

print(X, X.shape)  # (150, 4)
print(y, y.shape)  # (150, 1)
