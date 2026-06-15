import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 🧬 Speed 基于物理速度约束的清洗核心类
# ==========================================
class SpeedConstraintCleaner:
    def __init__(self, s_max, s_min):
        """
        :param s_max: 允许的最大正向变化速度 (上限)
        :param s_min: 允许的最大负向变化速度 (下限)
        """
        self.s_max = s_max
        self.s_min = s_min

    def fit_and_clean(self, series_data, timestamps):
        """
        基于速度物理边界约束进行时序清洗
        """
        n_samples = len(series_data)
        cleaned_series = np.array(series_data, dtype=float)

        # 记录每个点是否为合法点（True为正常，False为异常）
        labels = np.ones(n_samples, dtype=bool)

        # 1. 第一遍扫描：基于一阶导数（速度）识别不合理的突变点
        last_valid_idx = 0
        for i in range(1, n_samples):
            delta_t = timestamps[i] - timestamps[last_valid_idx]

            # 防止时间差为0导致除零异常
            if delta_t <= 0:
                delta_t = 1e-5

            delta_v = cleaned_series[i] - cleaned_series[last_valid_idx]
            current_speed = delta_v / delta_t

            # 物理边界审查
            if not (self.s_min <= current_speed <= self.s_max):
                labels[i] = False  # 判定为速度超限异常
            else:
                last_valid_idx = i  # 只有当前点合法，才更新为下一个比对的基准点

        # 2. 第二遍扫描：对识别出的异常点进行速度投影与局域插值修复
        for i in range(n_samples):
            if not labels[i]:
                # 寻找左侧最近的正常点
                left_idx = i - 1
                while left_idx >= 0 and not labels[left_idx]:
                    left_idx -= 1

                # 寻找右侧最近的正常点
                right_idx = i + 1
                while right_idx < n_samples and not labels[right_idx]:
                    right_idx += 1

                # 局域时序插值修复
                if left_idx >= 0 and right_idx < n_samples:
                    # 夹在两个正常点中间，使用线性插值恢复（即保持平稳速度过波）
                    t_l, t_r = timestamps[left_idx], timestamps[right_idx]
                    v_l, v_r = cleaned_series[left_idx], cleaned_series[right_idx]
                    ratio = (timestamps[i] - t_l) / (t_r - t_l)
                    cleaned_series[i] = v_l + ratio * (v_r - v_l)
                elif left_idx >= 0:
                    # 右侧没了，延续左侧最后一步的平稳期望
                    cleaned_series[i] = cleaned_series[left_idx]
                elif right_idx < n_samples:
                    # 左侧没了，回溯右侧常态
                    cleaned_series[i] = cleaned_series[right_idx]

        return cleaned_series


# ==========================================
# 🔌 ★ 统一模块化标准接口定义 ★
# ==========================================
def clean_data(df_dirty: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    符合 ChooseClean 资产库规范的统一 Speed 速度约束清洗接口

    参数:
        df_dirty (pd.DataFrame): 传入的脏数据
        ratio (float): 清洗比例控制 (0.0 ~ 1.0)。
                       ratio 趋近于 0：物理速度边界放得极宽，几乎不判定异常。
                       ratio 趋近于 1：物理速度边界剧烈收紧，稍微有突变刺毛立刻进行熨平修复。
    """
    df_result = df_dirty.copy()
    if ratio <= 0:
        return df_result

    # 1. 自动提取所有数值型特征列
    feature_cols = [c for c in df_result.columns if pd.api.types.is_numeric_dtype(df_result[c])]

    # 2. 获取或构造时间戳轴（算法计算速度必须依赖时间差 delta_t）
    time_col = None
    for col in ['timestamp', 'time', '时间']:
        if col in df_result.columns:
            time_col = col
            break
    if time_col:
        timestamps = pd.to_numeric(df_result[time_col], errors='coerce').fillna(0).values
    else:
        timestamps = np.arange(len(df_result), dtype=float)

    # 3. 预处理异常占位符
    error_placeholders = ["ERROR", "MISSING", "未知", "NaN", "nan"]
    for col in feature_cols:
        df_result[col] = df_result[col].replace(error_placeholders, np.nan)
        df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
        df_result[col] = df_result[col].fillna(df_result[col].mean() if not df_result[col].mean() is np.nan else 0.0)

    # 4. 循环遍历各列，自适应计算物理速度边界并执行清洗
    for col in feature_cols:
        series_data = df_result[col].values

        # 启发式自适应速度边界估计：利用一阶差分的标准差来捕捉当前列的常态基准速度
        diff_v = np.diff(series_data)
        diff_t = np.diff(timestamps)
        diff_t[diff_t == 0] = 1e-5
        speeds = diff_v / diff_t

        # 计算全局常态速度的标准差作为基准
        speed_std = np.std(speeds) if len(speeds) > 0 else 1.0
        if speed_std == 0: speed_std = 1.0

        # ★ 将外部 ratio 映射为速度边界的“收紧系数” ★
        # ratio 越大，边界收得越紧 (2.0 / ratio)
        boundary_multiplier = max(2.0 * (1.0 - ratio), 0.1)

        s_max = speed_std * boundary_multiplier
        s_min = -s_max

        # 5. 实例化并运行速度约束清洗器
        cleaner = SpeedConstraintCleaner(s_max=s_max, s_min=s_min)
        cleaned_col = cleaner.fit_and_clean(series_data, timestamps)

        # 写回数据
        df_result[col] = cleaned_col

    return df_result