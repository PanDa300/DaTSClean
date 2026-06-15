import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. 模型训练函数
# ==========================================
def train_and_save_model(train_csv_path, target_col, model_save_path='loss_rate_model.pkl'):
    """
    读取历史数据并训练预测模型

    参数:
        train_csv_path: 包含历史数据的CSV文件路径
        target_col: 你的CSV中代表“损失率”的那一列的列名
        model_save_path: 训练好的模型保存路径
    """
    print(f"📥 正在读取训练数据: {train_csv_path}...")
    try:
        df = pd.read_csv(train_csv_path)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

    # 简单的数据清洗：删除目标列为空的行，填补其他特征的缺失值为0
    df = df.dropna(subset=[target_col])
    df = df.fillna(0)

    # 如果有非数字的文本列（如类别），自动将其转换为数字编码 (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)

    # 提取特征 (X) 和目标变量 (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 记录特征列名，保证预测时列的顺序一致
    feature_names = X.columns.tolist()

    # 将数据划分为训练集 (80%) 和测试集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 正在训练随机森林回归模型...")
    # 初始化模型，100棵树
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 模型评估结果:")
    print(f"   平均绝对误差 (MAE): {mae:.4f} (预测值与真实值平均相差多少)")
    print(f"   决定系数 (R²):     {r2:.4f} (越接近1说明模型越准确)")

    # 保存模型和特征名
    model_data = {
        'model': model,
        'features': feature_names
    }
    joblib.dump(model_data, model_save_path)
    print(f"\n💾 模型已保存至: {model_save_path}")

    return model_data


# ==========================================
# 2. 数据预测函数
# ==========================================
def predict_new_data(new_csv_path, model_save_path='loss_rate_model.pkl'):
    """
    加载训练好的模型，对新的CSV数据进行预测
    """
    print(f"\n🔍 加载模型准备预测...")
    try:
        model_data = joblib.load(model_save_path)
        model = model_data['model']
        trained_features = model_data['features']
    except FileNotFoundError:
        print(f"❌ 找不到模型文件 {model_save_path}，请先训练模型！")
        return

    print(f"📥 正在读取新数据: {new_csv_path}...")
    df_new = pd.read_csv(new_csv_path)

    # 保存原始数据用于最后合并结果
    df_result = df_new.copy()

    # 处理新数据：转换为数字编码，填补缺失值
    df_new = pd.get_dummies(df_new)
    df_new = df_new.fillna(0)

    # ★ 关键步骤：对齐列名 ★
    # 确保新数据拥有和训练数据完全一样的列，缺失的列补0
    for col in trained_features:
        if col not in df_new.columns:
            df_new[col] = 0

    # 只保留训练时用到的列，并保持相同顺序
    X_new = df_new[trained_features]

    # 执行预测
    print("🔮 正在计算预测损失率...")
    predictions = model.predict(X_new)

    # 将预测结果作为一个新列添加到原始表格中
    df_result['预测损失率'] = predictions

    # 保存预测结果
    output_path = new_csv_path.replace('.csv', '_predicted.csv')
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 预测完成！结果已保存至: {output_path}")
    return predictions[0]


# ==========================================
# 💡 测试与使用说明
# ==========================================
if __name__ == "__main__":
    import os

    # --- 这里我为你生成一些测试用的假数据 ---
    # 真实使用时，请直接替换为你自己的 CSV 文件路径
    if not os.path.exists('historical_data.csv'):
        print("创建测试用的历史数据 'historical_data.csv'...")
        test_data = pd.DataFrame({
            '温度': np.random.uniform(20, 35, 100),
            '湿度': np.random.uniform(40, 80, 100),
            '机器型号': np.random.choice(['A型', 'B型', 'C型'], 100),
            '运行时间': np.random.uniform(10, 1000, 100),
            '损失率': np.random.uniform(0.01, 0.15, 100)  # 目标列
        })
        test_data.to_csv('historical_data.csv', index=False)

    if not os.path.exists('new_data.csv'):
        print("创建测试用的待预测数据 'new_data.csv'...")
        new_data = pd.DataFrame({
            '温度': [30.5, 22.1],
            '湿度': [75.0, 45.2],
            '机器型号': ['A型', 'B型'],
            '运行时间': [850, 120]
            # 注意：新数据里没有“损失率”这一列，我们需要预测它
        })
        new_data.to_csv('new_data.csv', index=False)
    # ------------------------------------------

    # 步骤 1: 使用历史数据训练模型
    # 请把 '历史数据.csv' 和 '损失率' 换成你实际的文件名和列名
    train_and_save_model(
        train_csv_path='historical_data.csv',
        target_col='损失率'
    )

    # 步骤 2: 输入新的CSV文件，输出带有预测结果的新CSV
    predict_new_data(
        new_csv_path='new_data.csv'
    )