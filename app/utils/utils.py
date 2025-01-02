import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算MAPE
    epsilon = 1e-10  # 很小的数
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # 如果MAPE不合理，使用另一种计算方法
    if mape > 100 or np.isnan(mape):
        mape = np.mean(np.abs(y_true - y_pred) / np.mean(y_true)) * 100
    
    return mape, rmse, mae
