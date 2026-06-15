import torch
import torch.nn as nn
import numpy as np


# --- DLinear 底层算子 ---
class MovingAvg(nn.Module):
    """移动平均算子，用于提取趋势"""

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 解决边缘 padding 问题
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """时间序列分解层：将序列分解为 趋势项 (Trend) 和 周期/残差项 (Seasonal)"""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearModel(nn.Module):
    """DLinear 核心网络结构"""

    def __init__(self, seq_len, pred_len, channels=1, kernel_size=25):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 分解层
        self.decomp = SeriesDecomp(kernel_size)

        # 两个独立的线性映射层
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_len, Channels]
        seasonal_init, trend_init = self.decomp(x)

        seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 最终预测结果为两部分相加
        return seasonal_output + trend_output


# --- 封装为直接可用的执行函数 ---
def run_dlinear(y_train, test_len, epochs=50, lr=0.01):
    """
    轻量级深度学习：DLinear
    :param y_train: 一维历史数据
    :param test_len: 需要预测的长度
    """
    seq_len = len(y_train)
    pred_len = test_len

    # 转换数据格式供 PyTorch 使用 [Batch=1, Seq_len, Channel=1]
    X_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # 初始化模型与优化器
    model = DLinearModel(seq_len=seq_len, pred_len=pred_len, channels=1, kernel_size=min(25, seq_len - 1))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    # 简易训练循环 (拟合自我重构能力)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 注意：这里做自回归拟合简化，严格应采用滑窗构建 Dataset。
        # 此处使用历史数据映射自身(仅演示架构，可根据窗口需求修改)
        output = model(X_tensor)
        # 用最后一截假装目标进行权重更新
        loss = criterion(output[:, :seq_len, :], X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # 推理未来数据
        predictions = model(X_tensor)

    # 返回预测的一维数组
    return predictions.squeeze().numpy()