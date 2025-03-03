import numpy as np

def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # 避免除以零和异常值
    epsilon = 1e-10  # 添加小的常数避免除以零
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return min(mape, 100)  # 限制最大误差为100%

def calculate_rmse(y_true, y_pred):
    """计算相对均方根误差"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # 计算相对误差
    epsilon = 1e-10  # 添加小的常数避免除以零
    relative_errors = (y_true - y_pred) / (y_true + epsilon)
    
    return np.sqrt(np.mean(relative_errors ** 2)) * 100

def calculate_mae(y_true, y_pred):
    """计算相对平均绝对误差"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # 计算相对误差
    epsilon = 1e-10  # 添加小的常数避免除以零
    relative_errors = np.abs((y_true - y_pred) / (y_true + epsilon))
    
    return np.mean(relative_errors) * 100

def calculate_metrics(y_true, y_pred):
    """计算所有评估指标
    
    返回值:
    - MAPE: 平均绝对百分比误差 (%)
    - RMSE: 相对均方根误差 (%)
    - MAE: 相对平均绝对误差 (%)
    """
    return (
        calculate_mape(y_true, y_pred),
        calculate_rmse(y_true, y_pred),
        calculate_mae(y_true, y_pred)
    )
