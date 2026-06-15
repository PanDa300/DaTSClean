import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 🧬 Smooth 自适应时序平滑清洗核心类
# ==========================================
class AdaptiveExponentialSmoothing:
    def __init__(self, alpha=0.3):
        """
        :param alpha: 平滑系数 (0 < alpha <= 1)。
                      alpha 越小，平滑后曲线越平健，对高频毛刺滤除越强；
                      alpha 越大，越贴近原始观测值。
        """
        self.alpha = alpha
        self.smooth_val = None
        self.trend = 0.0  # 引入一阶趋势项，防止平滑后产生严重相位滞后

    def update(self, measurement, is_nan=False):
        """
        单步自适应平滑更新
        """
        if self.smooth_val is None:
            # 初始化状态
            self.smooth_val = measurement if not is_nan else 0.0
            return self.smooth_val

        if is_nan:
            # 遭遇损坏/缺失点：完全由历史趋势向前推算进行插值填补
            predicted_val = self.smooth_val + self.trend
            # 更新状态为预测值
            self.smooth_val = predicted_val
            return self.smooth_val

        # 正常点：执行二阶平滑更新（Holt线性平滑核心思想）
        last_smooth = self.smooth_val
        # 1. 计算当前的趋势平滑值
        self.smooth_val = self.alpha * measurement + (1 - self.alpha) * (last_smooth + self.trend)
        # 2. 动态更新趋势增量项 (防止滞后)
        self.trend = 0.1 * (self.smooth_val - last_smooth) + 0.9 * self.trend

        return self.smooth_val


# ==========================================
# 🔌 ★ 统一模块化标准接口定义 ★
# ==========================================
def clean_data(df_dirty: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    符合 ChooseClean 资产库规范的统一 Smooth 平滑清洗接口

    参数:
        df_dirty (pd.DataFrame): 传入的脏数据
        ratio (float): 清洗比例控制 (0.0 ~ 1.0)。
                       ratio 趋近于 0：完全保留噪声毛刺，不执行任何趋势收拢。
                       ratio 趋近于 1：激进消除高频扰动，完全依赖长期平滑趋势修复。
    """
    df_result = df_dirty.copy()
    if ratio <= 0:
        return df_result

    # 1. 自动提取所有数值型特征列
    feature_cols = [c for c in df_result.columns if pd.api.types.is_numeric_dtype(df_result[c])]

    # 2. 将外部资源清洗比例 ratio 映射为平滑系数 alpha
    # ratio 越大（资源投入多），alpha 越小，说明对噪声容忍度越低，要求平滑过滤的力度越彻底
    # 当 ratio=0.1 时，alpha约为 0.9（轻微平滑）；当 ratio=0.9 时，alpha约为 0.1（极强平滑）
    alpha_adjusted = max(1.0 - ratio, 0.05)

    # 3. 循环遍历各列，对每一列特征分别应用自适应平滑清洗
    for col in feature_cols:
        series_data = df_result[col].values
        n_samples = len(series_data)

        # 寻找第一个非空有效的初值
        first_valid_idx = 0
        for idx, val in enumerate(series_data):
            if not pd.isna(val) and val != "ERROR" and val != "":
                first_valid_idx = idx
                break

        init_x = float(series_data[first_valid_idx]) if first_valid_idx < n_samples else 0.0
        if np.isnan(init_x):
            init_x = 0.0

        # 初始化平滑清洗器
        cleaner = AdaptiveExponentialSmoothing(alpha=alpha_adjusted)
        cleaner.smooth_val = init_x

        cleaned_series = np.zeros(n_samples)
        error_placeholders = ["ERROR", "MISSING", "未知", "NaN", "nan"]

        # 4. 执行一维时间序列扫描
        for i in range(n_samples):
            val = series_data[i]

            # 判断当前数据行是否已被污染
            is_corrupted = pd.isna(val) or (str(val) in error_placeholders)

            if is_corrupted:
                # 缺失突变点：由趋势线直接预测拉回
                filtered_val = cleaner.update(0.0, is_nan=True)
            else:
                # 常态数据流：执行滤波去噪平滑
                filtered_val = cleaner.update(float(val), is_nan=False)

            cleaned_series[i] = filtered_val

        # 5. 回填清洗后的平滑时序特征列
        df_result[col] = cleaned_series

    return df_result