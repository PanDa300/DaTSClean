import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def inject_errors_to_csv(input_csv, error_rate_a, noise_scale=3.0, target_col='损失率'):
    """
    向 CSV 文件中的数值型列注入高斯（正态分布）误差。

    参数:
        input_csv (str): 原始纯净数据文件路径
        error_rate_a (float): 注入误差的数据比例 (0.0 ~ 1.0)
        noise_scale (float): 噪声强度系数。默认 3.0 表示生成标准差 3 倍级别的高斯噪声，以制造明显异常。
        target_col (str): 下游任务的标签列（建议保护标签列不加噪，以免影响最终模型评估的基准）
    """
    # 1. 读取数据并设置随机种子保证实验可复现
    df = pd.read_csv(input_csv)
    np.random.seed(42)

    # 2. 筛出所有的数值型列
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # 保护下游预测的标签列不被污染（极度推荐）
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if not numeric_cols:
        print("⚠️ 未找到可注入数值噪声的数值列！")
        return

    # 3. 计算需要加噪的总单元格数量
    total_cells = len(df) * len(numeric_cols)
    num_errors = int(total_cells * error_rate_a)

    # 如果错误率为 0，直接另存文件返回
    output_csv = os.path.join(
        os.path.dirname(input_csv),
        f"{os.path.splitext(os.path.basename(input_csv))[0]}_{error_rate_a}.csv"
    )

    if num_errors == 0:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return

    # 4. 随机生成要注入噪声的二维坐标 (行索引, 列名)
    row_indices = np.random.randint(0, len(df), size=num_errors)
    col_indices = np.random.choice(numeric_cols, size=num_errors)

    # 提前计算各列的标准差，作为该列高斯分布的尺度 \sigma
    col_stds = {}
    for col in numeric_cols:
        std = df[col].std()
        # 如果某列全是常数(方差为0)或全空，赋予一个微小的默认抖动量
        col_stds[col] = std if pd.notna(std) and std != 0 else 1.0

    # 5. 循环注入高斯噪声
    for r_idx, c_name in zip(row_indices, col_indices):
        c_std = col_stds[c_name]

        # 核心数学公式：生成正态分布噪声 N(0, (\sigma * scale)^2)
        noise = np.random.normal(loc=0.0, scale=c_std * noise_scale)

        original_val = df.at[r_idx, c_name]

        # 如果该点原本就是空值，直接用噪声替代；否则在原值上叠加扰动
        if pd.isna(original_val):
            df.at[r_idx, c_name] = noise
        else:
            df.at[r_idx, c_name] = float(original_val) + noise

    # 6. 保存为脏数据文件
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')