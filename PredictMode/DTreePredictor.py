from sklearn.tree import DecisionTreeRegressor
import numpy as np


def run_decision_tree(X_train, y_train, X_test):
    """
    传统机器学习：决策树回归
    """
    # 初始化决策树回归器，固定 random_state 保证可复现
    # 限制最大深度防止在小数据集上过拟合
    model = DecisionTreeRegressor(max_depth=10, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测未来结果
    y_pred = model.predict(X_test)

    return y_pred