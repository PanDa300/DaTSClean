import pandas as pd
import numpy as np
import warnings

# 忽略 pandas 的某些填补警告
warnings.filterwarnings('ignore')


def clean_data(df_dirty, ratio):
    """
    基础数据修复算法 (BaseRepair)

    统一接口要求：
    - 输入：
        df_dirty (pd.DataFrame): 包含噪音或缺失值的脏数据
        ratio (float): 清洗比例 (0.0 ~ 1.0)。
                       决定了数据集中有多大比例的行会被实施彻底修复，其余保持原样。
    - 输出：
        df_cleaned (pd.DataFrame): 处理后的数据
    """

    # 1. 拷贝数据，避免直接修改原始对象
    df_result = df_dirty.copy()

    # 如果比例为 0，直接返回原数据
    if ratio <= 0:
        return df_result

    # 2. 根据 ratio 抽取需要清洗的行
    # 这里采用 random_state=42 保证可重复性，方便你在不同算法间做公平对比
    df_to_clean = df_result.sample(frac=ratio, random_state=42)

    # 获取需要清洗的行的索引
    clean_indices = df_to_clean.index

    # 3. 预处理：将常见的文本型错误标识转换为标准的 np.nan，方便统一处理
    error_placeholders = ["ERROR", "MISSING", "未知", "NaN", "nan", ""]
    df_to_clean.replace(error_placeholders, np.nan, inplace=True)

    # 4. 自动根据列的数据类型进行基础修复 (中位数/众数)
    for col in df_to_clean.columns:
        # 提取当前需要清洗的那一部分列数据
        col_data = df_to_clean[col]

        # 判断是否为数值型列 (int 或 float)
        if pd.api.types.is_numeric_dtype(col_data):
            # 计算中位数
            median_val = col_data.median()
            # 如果整列都是空的（无法计算中位数），则默认填 0；否则填中位数
            if pd.isna(median_val):
                df_to_clean[col] = col_data.fillna(0)
            else:
                df_to_clean[col] = col_data.fillna(median_val)

        # 处理文本/类别型列 (object 或 string)
        else:
            # 计算众数 (出现次数最多的值)
            mode_vals = col_data.mode()
            if len(mode_vals) > 0:
                df_to_clean[col] = col_data.fillna(mode_vals[0])
            else:
                # 如果整列全空，填补为通用标识
                df_to_clean[col] = col_data.fillna("Unknown")

    # 5. 将清洗好的部分拼接回原数据
    # 使用 .loc 直接按索引覆盖原数据中对应的行
    df_result.loc[clean_indices] = df_to_clean

    return df_result