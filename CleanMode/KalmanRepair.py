import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 🚀 极速版卡尔曼时序滤波修复算子
# ==========================================
def clean_data(df_dirty, clean_ratio=1.0):
    """
    使用卡尔曼滤波进行时序数据修复 (针对百万级循环进行底层 NumPy 加速)
    """
    df_result = df_dirty.copy()

    # 提取可能是数值或者包含 "ERROR" 字符串的列
    numeric_cols = df_result.select_dtypes(include=[np.number, 'object']).columns.tolist()

    n_samples = len(df_result)
    clean_limit = int(n_samples * clean_ratio)

    # 如果清洗比例为 0，直接返回脏数据
    if clean_limit == 0:
        return df_result

    for col in numeric_cols:
        # 1. 🌟 极限加速：用 Pandas 底层 C 引擎一次性清洗所有异常字符串
        # 这一步将所有的 "ERROR", "MISSING", "未知" 等文本瞬间变成标准的 np.nan
        float_arr = pd.to_numeric(df_result[col], errors='coerce').values

        # 如果这个列全是 nan，或者根本不是数值列，跳过
        if np.isnan(float_arr).all():
            continue

        # 2. 预先计算所有掩码和方差（向量化极速操作）
        is_missing_arr = np.isnan(float_arr)
        col_std = np.nanstd(float_arr)

        # 卡尔曼滤波参数初始化
        process_noise_Q = 1e-4 if col_std == 0 else (col_std ** 2) * 1e-3
        R_adjusted = 1e-2  # 假定测量噪声为固定值（你可根据原版自行调整）

        # 寻找第一个非 nan 的值作为滤波器初始状态
        valid_indices = np.where(~is_missing_arr)[0]
        if len(valid_indices) == 0:
            continue
        init_x = float_arr[valid_indices[0]]

        # 3. 🌟 纯数值 Python 极速循环
        # 注意：这里去掉了所有 pd.isna() 和 in 列表的查询操作，纯算浮点数，速度提升百倍！
        P = 1.0
        x = init_x
        Q = process_noise_Q
        R = R_adjusted

        cleaned_series = float_arr.copy()

        for i in range(clean_limit):
            p_minus = P + Q
            if is_missing_arr[i]:
                P = p_minus
                cleaned_series[i] = x  # 遇到缺失，用系统预测值填补
            else:
                measurement = float_arr[i]
                K = p_minus / (p_minus + R)
                x = x + K * (measurement - x)
                P = (1 - K) * p_minus
                cleaned_series[i] = x

        # 4. 替换回原 DataFrame（只替换前 clean_limit 的数据，保持原有索引绝对安全）
        if clean_limit == n_samples:
            df_result[col] = cleaned_series
        else:
            df_result.loc[df_result.index[:clean_limit], col] = cleaned_series[:clean_limit]

    return df_result