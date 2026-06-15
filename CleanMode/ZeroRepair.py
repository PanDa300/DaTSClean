import pandas as pd
import numpy as np


def clean_data(df_dirty, ratio):
    """粗暴补零法：所有的缺失值全部填 0"""
    df_result = df_dirty.copy()
    if ratio <= 0: return df_result

    df_to_clean = df_result.sample(frac=ratio, random_state=42)
    clean_indices = df_to_clean.index

    error_placeholders = ["ERROR", "MISSING", "未知", "NaN", "nan", ""]
    df_to_clean.replace(error_placeholders, np.nan, inplace=True)

    # 不管什么类型，简单粗暴全部填 0
    df_to_clean = df_to_clean.fillna(0)

    df_result.loc[clean_indices] = df_to_clean
    return df_result