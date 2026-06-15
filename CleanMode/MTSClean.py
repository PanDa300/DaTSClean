import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 🧬 MTSClean 多维属性时序清洗核心类
# ==========================================
class MultiAttributeTimeSeriesCleaner:
    def __init__(self, window_size=5):
        """
        MTSClean 核心框架简化增强版
        :param window_size: 局部时间时序连续性审查的窗口大小
        """
        self.window_size = window_size

    def _compute_mahalanobis_scores(self, matrix_values):
        """
        利用协方差矩阵计算各行在多维空间中的马氏距离得分
        得分越高，说明该多维空间点越偏离多维联合分布（即异常程度越高）
        """
        N, D = matrix_values.shape
        # 计算全局均值和协方差矩阵
        mean_vec = np.mean(matrix_values, axis=0)
        cov_matrix = np.cov(matrix_values, rowvar=False)

        # 防止协方差矩阵奇异（不可逆），加入微小的正则化扰动
        if D > 1:
            cov_matrix += np.eye(D) * 1e-5
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        else:
            inv_cov_matrix = np.array([[1.0 / max(cov_matrix, 1e-5)]])

        scores = np.zeros(N)
        for i in range(N):
            delta = matrix_values[i, :] - mean_vec
            # 马氏距离公式: d = sqrt( delta * Inv(Cov) * delta^T )
            scores[i] = np.sqrt(np.dot(np.dot(delta, inv_cov_matrix), delta.T))
        return scores

    def fit_and_clean(self, matrix_values, ratio):
        """
        多维联合清洗核心逻辑
        """
        N, D = matrix_values.shape
        cleaned_matrix = matrix_values.copy()

        # 1. 计算多维空间异常空间马氏得分
        scores = self._compute_mahalanobis_scores(matrix_values)

        # 2. 根据给定的清洗比例 ratio 分配预算，筛选出最需要清洗的行索引
        budget = max(int(N * ratio), 1)
        target_indices = np.argsort(scores)[-budget:]  # 提取得分最高（最异常）的前 budget 个点

        # 3. 计算多维属性间的回归关联权重（用于利用其他正常维度来修复损坏维度）
        # 简化实现：利用局部滑动时间窗建立时序连续性期望，并结合全局属性均值做加权投影修复
        global_mean = np.mean(matrix_values, axis=0)

        for idx in target_indices:
            # 确定当前点周围的局部常态上下文窗口
            start_w = max(0, idx - self.window_size)
            end_w = min(N, idx + self.window_size + 1)

            # 局部时序期望值
            local_context = matrix_values[start_w:end_w, :]
            local_reconstructed_expected = np.mean(local_context, axis=0)

            # 找出当前行中偏离局部期望最严重的维度（被污染的属性）
            row_data = matrix_values[idx, :]
            dimension_residuals = np.abs(row_data - local_reconstructed_expected)
            worst_dimension = np.argmax(dimension_residuals)

            # 基于 MTSClean 空间多维投影恢复思想：
            # 被污染维度的修复值 = 局部时序期望值 + 全局多维属性关联偏置修正
            repaired_value = local_reconstructed_expected[worst_dimension] * 0.7 + global_mean[worst_dimension] * 0.3

            # 执行针对性修复
            cleaned_matrix[idx, worst_dimension] = repaired_value

        return cleaned_matrix


# ==========================================
# 🔌 ★ 统一模块化标准接口定义 ★
# ==========================================
def clean_data(df_dirty: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    符合 ChooseClean 资产库规范的统一 MTSClean 清洗接口

    参数:
        df_dirty (pd.DataFrame): 传入的多维脏数据
        ratio (float): 清洗比例控制 (0.0 ~ 1.0)。等价转换为多维空间联合残差的控制清洗阈值。
    """
    df_result = df_dirty.copy()
    if ratio <= 0:
        return df_result

    # 1. 自动提取所有数值型特征列（多维属性空间）
    feature_cols = [c for c in df_result.columns if pd.api.types.is_numeric_dtype(df_result[c])]

    if not feature_cols:
        return df_result

    # 2. 数据矩阵提取与初步异常占位符清洗（将 NaN 或错误字符预填补为列均值，防止矩阵运算崩溃）
    error_placeholders = ["ERROR", "MISSING", "未知", "NaN", "nan"]
    for col in feature_cols:
        df_result[col] = df_result[col].replace(error_placeholders, np.nan)
        # 用均值做初步填充，保证协方差矩阵计算不出现 NaN
        df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
        df_result[col] = df_result[col].fillna(df_result[col].mean() if not df_result[col].mean() is np.nan else 0.0)

    matrix_values = df_result[feature_cols].values

    # 3. 实例化 MTSClean 核心清洗器
    cleaner = MultiAttributeTimeSeriesCleaner(window_size=5)

    # 4. 执行多维空间时序联合修复
    cleaned_matrix = cleaner.fit_and_clean(matrix_values, ratio)

    # 5. 将清洗修复后的矩阵无缝写回 Pandas DataFrame 结构
    df_result[feature_cols] = cleaned_matrix

    return df_result