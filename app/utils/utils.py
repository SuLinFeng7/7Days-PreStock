import numpy as np

def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_rmse(y_true, y_pred):
    """计算均方根误差"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    """计算平均绝对误差"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred))

def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    return (
        calculate_mape(y_true, y_pred),
        calculate_rmse(y_true, y_pred),
        calculate_mae(y_true, y_pred)
    )
