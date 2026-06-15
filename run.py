import time
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')


def evaluate_cleaning_algorithm(
        dirty_data_path,
        clean_ratio_b,
        clean_func,
        algo_name="指定算法",
        target_col='损失率',
        model_path='loss_rate_model.pkl'
):
    print("-" * 60)
    print(f"📊 正在评估算法: 【{algo_name}】 | 清洗比例 b = {clean_ratio_b}")
    print("-" * 60)

    # 1. 读取脏数据
    df_dirty = pd.read_csv(dirty_data_path)

    # 🚨 【容错核心】: 检查并保护标签列
    if target_col not in df_dirty.columns:
        print(f"⚠️ 警告: 脏数据中未找到标签列 '{target_col}'！")
        print(f"当前可用的列名有: {df_dirty.columns.tolist()}")
        # 如果真的找不到，尝试从原始文件里补救，或者直接报错提示
        raise KeyError(f"数据集中缺失目标列: '{target_col}'，请检查原始数据。")

    # 强行把真实的标签列先剥离出来保存，防止被后面的清洗算法误删或破坏
    y_dirty_backup = df_dirty[target_col].copy()

    # 加载模型
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        trained_features = model_data['features']
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 找不到模型文件 {model_path}，请先训练模型！")

    # 2. 🟢 启动计时
    start_time = time.time()

    # 执行清洗
    try:
        df_cleaned = clean_func(df_dirty.copy(), clean_ratio_b)
    except Exception as e:
        print(f"   ❌ 算法 [{algo_name}] 在清洗阶段崩溃: {e}")
        return None

    # 🚨 【容错核心】: 清洗完后，把刚才备份的标签列强行塞回去，确保特征对齐时不出错
    df_cleaned[target_col] = y_dirty_backup.values

    # 3. 对齐下游模型的特征工程
    df_new = pd.get_dummies(df_cleaned)
    df_final = pd.DataFrame(0, index=np.arange(len(df_new)), columns=trained_features)
    for col in trained_features:
        if col in df_new.columns:
            df_final[col] = df_new[col].values
    df_final = df_final.fillna(0)

    # 下游模型预测
    y_pred = model.predict(df_final)

    # 🔴 停止计时
    elapsed_time = time.time() - start_time

    # 4. 读取最原始的纯净数据计算真实误差
    origin_csv = dirty_data_path.split('_0.')[0] + '.csv'
    if not os.path.exists(origin_csv):
        origin_csv = 'historical_data.csv'

    df_original = pd.read_csv(origin_csv)
    y_true = df_original[target_col].fillna(0)

    # 5. 计算全套评价指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    y_true_clipped = np.clip(y_true, 0, None)
    y_pred_clipped = np.clip(y_pred, 0, None)
    try:
        rmsle = np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))
    except:
        rmsle = np.nan

    metrics_x10 = {
        "算法名称": algo_name,
        "清洗比例": clean_ratio_b,
        "MSE (x10)": mse * 10,
        "RMSE (x10)": rmse * 10,
        "MAE (x10)": mae * 10,
        "MAPE (x10)": mape * 10,
        "RMSLE (x10)": rmsle * 10 if not np.isnan(rmsle) else np.nan,
        "总耗时_秒 (x10)": elapsed_time * 10
    }

    print(f"   📉 MSE (x10)   : {metrics_x10['MSE (x10)']:.6f}")
    print(f"   📉 RMSE (x10)  : {metrics_x10['RMSE (x10)']:.6f}")
    print(f"   ⏳ 耗时 (x10)  : {metrics_x10['总耗时_秒 (x10)']:.4f} 秒\n")

    return metrics_x10

# ==========================================
# 💡 如何多算法批量对比使用示例
# ==========================================
if __name__ == "__main__":
    import os

    # 1. 导入你的两个清洗算法
    from CleanMode.BaseRepair import clean_data as base_repair
    from CleanMode.SHoTClean import clean_data as shot_clean

    # 2. 准备脏数据路径 (请确保该文件已存在)
    dirty_data = "my_dataset_0.15.csv"

    if os.path.exists(dirty_data):
        # 测试 BaseRepair
        base_res = evaluate_cleaning_algorithm(
            dirty_data_path=dirty_data,
            clean_ratio_b=0.5,
            clean_func=base_repair,
            algo_name="BaseRepair_中位数众数"
        )

        # 测试新改造的 SHoTClean
        shot_res = evaluate_cleaning_algorithm(
            dirty_data_path=dirty_data,
            clean_ratio_b=0.5,
            clean_func=shot_clean,
            algo_name="SHoTClean_时序动态规划"
        )

        # 3. 打印对比表格
        df_compare = pd.DataFrame([base_res, shot_res])
        print("====== 📊 算法横向对比矩阵 (已放大10倍) ======")
        print(df_compare.to_markdown(index=False))