from statsmodels.tsa.arima.model import ARIMA
import warnings


def run_arima(y_train, test_len, order=(5, 1, 0)):
    """
    传统统计学：ARIMA 模型
    :param y_train: 一维历史时序数据 (目标列)
    :param test_len: 需要预测的未来步长长度
    :param order: (p, d, q) 参数。默认为 (5,1,0) 适合带有一定趋势的数据
    """
    # 忽略 statsmodels 底层的收敛警告
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        # 构建并拟合 ARIMA 模型
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()

        # 预测未来 test_len 步
        y_pred = model_fit.forecast(steps=test_len)
    except Exception as e:
        # ARIMA 对极端脏数据有时会崩溃，这里做个兜底
        print(f"ARIMA 拟合失败: {e}")
        y_pred = [y_train[-1]] * test_len  # 降级为 Naive 预测（复制最后一天）

    return y_pred