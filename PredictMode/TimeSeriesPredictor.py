import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    mean_absolute_percentage_error
)
import warnings

warnings.filterwarnings('ignore')


def _create_sliding_window(df, target_col, window_size=5):
    """
    [内部方法] 将二维表格数据转换为时序预测的滞后特征（Lag Features）。
    """
    numeric_df = df.select_dtypes(include=[np.number]).fillna(method='ffill').fillna(0)
    data = numeric_df.values

    if target_col not in numeric_df.columns:
        target_idx = -1
    else:
        target_idx = numeric_df.columns.get_loc(target_col)

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size].flatten())
        y.append(data[i + window_size, target_idx])

    return np.array(X), np.array(y)


def evaluate_downstream_task(df_cleaned, df_original, target_col, test_ratio=0.2, window_size=5):
    """
    标准化的下游时序预测评估接口。
    前 (1-test_ratio) 的清洗数据用于训练，预测剩下的 test_ratio 数据，并与原数据对比。

    参数:
        df_cleaned (pd.DataFrame): 经过算法清洗后的数据
        df_original (pd.DataFrame): 最原始纯净的数据（用作 Ground Truth 对比）
        target_col (str): 需要预测的目标列名
        test_ratio (float): 测试集所占的比例 (默认 0.2，即预测最后 20%)
        window_size (int): 时序滑动窗口的大小

    返回:
        dict: 包含 5 大误差指标的字典 (已自动放大 10 倍)
    """
    # 1. 构造特征与标签
    X_clean, y_clean = _create_sliding_window(df_cleaned, target_col, window_size)
    X_orig, y_orig = _create_sliding_window(df_original, target_col, window_size)

    # 2. 严格按时间顺序切割数据集
    split_idx = int(len(X_clean) * (1 - test_ratio))

    # 训练集：使用清洗后的历史数据
    X_train = X_clean[:split_idx]
    y_train = y_clean[:split_idx]

    # 测试集：输入当前特征，预测未来
    X_test = X_clean[split_idx:]

    # ★ 核心：测试集的标准答案永远是原始干净数据
    y_test_true = y_orig[split_idx:]

    # 3. 现场训练下游预测模型 (默认使用 Ridge 回归，速度快且稳定)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # 4. 执行滚动预测
    y_pred = model.predict(X_test)

    # 5. 计算误差指标
    mse = mean_squared_error(y_test_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true, y_pred)

    y_true_safe = np.where(y_test_true == 0, 1e-10, y_test_true)
    mape = mean_absolute_percentage_error(y_true_safe, y_pred)

    try:
        rmsle = np.sqrt(mean_squared_log_error(np.clip(y_test_true, 0, None), np.clip(y_pred, 0, None)))
    except ValueError:
        rmsle = np.nan

    # 返回统一放大 10 倍的指标字典
    return {
        "RMSE": rmse * 10,
        "MSE": mse * 10,
        "MAE": mae * 10,
        "RMSLE": rmsle * 10 if not np.isnan(rmsle) else np.nan,
        "MAPE": mape * 10
    }