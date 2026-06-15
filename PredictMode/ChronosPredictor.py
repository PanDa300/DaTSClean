import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error
)

# 引入 Chronos
from chronos.chronos2 import Chronos2Pipeline

warnings.filterwarnings('ignore')

print("⏳ 正在初始化 Chronos-2 大模型，请耐心等待 (此操作全局仅执行一次)...")
try:
    # 全局加载模型，避免在网格搜索中重复加载导致显存 OOM 或时间爆炸
    # 如果显存不够，可将 "amazon/chronos-2" 换为 "amazon/chronos-2-small" 或 "amazon/chronos-2-mini"
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print("✅ Chronos-2 大模型加载成功！")
except Exception as e:
    print(f"❌ Chronos-2 模型加载失败，请检查环境、网络或显存: {e}")
    pipeline = None


def evaluate_chronos_forecast(df_cleaned, df_original, target_col, test_ratio=0.2):
    """
    使用 Amazon Chronos-2 时序大模型进行下游任务预测评估。
    严格取前 (1-test_ratio) 作为 Context，预测后 test_ratio，并与原始纯净数据计算真实误差。
    """
    if pipeline is None:
        raise ValueError("Chronos Pipeline 未初始化，无法进行预测。")

    # 1. 确定训练与预测的切割点
    split_idx = int(len(df_cleaned) * (1 - test_ratio))
    prediction_length = len(df_cleaned) - split_idx

    # ==========================================
    # 2. 构造 Chronos 需要的标准时序格式
    # ==========================================
    df_chronos = df_cleaned.copy()

    # 强制注入虚拟的序列 ID 和连续时间戳 (Chronos API 强依赖这两个字段)
    df_chronos['id'] = 'ts_1'
    df_chronos['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df_chronos), freq='H')

    # 提取特征变量 (协变量 Covariates)
    features = [c for c in df_cleaned.columns if c != target_col]

    # 构造历史上下文 context_df (包含目标值和特征)
    context_df = df_chronos.iloc[:split_idx].copy()

    # 构造未来特征 future_df (必须剔除目标列 target_col)
    future_df = df_chronos.iloc[split_idx:].copy()
    future_df = future_df.drop(columns=[target_col])

    # ==========================================
    # 3. 调用 Chronos 执行预测
    # ==========================================
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=prediction_length,
        quantile_levels=[0.5],  # 我们只取 0.5 的分位数作为具体的点预测结果 (Point Forecast)
        id_column="id",
        timestamp_column="timestamp",
        target=target_col,
    )

    # 提取 Chronos 预测出的中位数结果数组
    y_pred = pred_df['0.5'].values

    # ==========================================
    # 4. 提取真实答案并计算工业级容错指标
    # ==========================================
    # 🎯 对比答案：永远使用【最原始、未加噪】的后 20% 数据作为 Ground Truth
    y_test_true = df_original.iloc[split_idx:][target_col].fillna(0).values

    mse = mean_squared_error(y_test_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true, y_pred)

    # 采用 WAPE 防除 0 崩溃
    sum_y_true = np.sum(np.abs(y_test_true))
    if sum_y_true == 0:
        mape = 0.0
    else:
        mape = np.sum(np.abs(y_test_true - y_pred)) / sum_y_true

    try:
        rmsle = np.sqrt(mean_squared_log_error(np.clip(y_test_true, 0, None), np.clip(y_pred, 0, None)))
    except Exception:
        rmsle = np.nan

    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "RMSLE": rmsle,
        "MAPE": mape
    }