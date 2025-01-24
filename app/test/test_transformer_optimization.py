import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.models.transformer_model import train_transformer_model, TimeSeriesTransformer
from app.data.data_preprocessing import get_stock_data, prepare_prediction_data
import torch
from app.config.model_versions import MODEL_VERSIONS
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

def progress_callback(epoch, total_epochs, train_loss=None, val_loss=None):
    """训练进度回调函数"""
    progress = (epoch / total_epochs) * 100
    if train_loss is not None and val_loss is not None:
        print(f"\rEpoch [{epoch}/{total_epochs}] {progress:.1f}% - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}", end="")
    else:
        print(f"\rEpoch [{epoch}/{total_epochs}] {progress:.1f}%", end="")
    if epoch == total_epochs:
        print()  # 打印换行

def test_transformer_model_optimization():
    """测试优化后的Transformer模型在比亚迪股票数据上的表现"""
    try:
        # 设置测试参数
        stock_code = "002594.SZ"  # 比亚迪股票代码
        prediction_days = 30      # 预测天数
        train_years = 10         # 训练年限
        
        # 计算日期范围
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*train_years)
        
        # 格式化日期
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"\n开始获取{stock_code}的历史数据...")
        print(f"数据范围: {start_date_str} 到 {end_date_str}")
        
        # 获取股票数据
        df = get_stock_data(stock_code, start_date_str, end_date_str)
        assert not df.empty, "获取股票数据失败"
        print(f"成功获取{len(df)}条历史数据")
        
        # 准备训练数据
        print("\n准备训练数据...")
        X_train, y_train, X_test, y_test, price_scaler, feature_scaler = prepare_prediction_data(df)
        
        # 确保数据维度正确
        assert len(X_train.shape) == 3, f"训练数据维度错误: {X_train.shape}"
        assert len(y_train.shape) == 1, f"训练标签维度错误: {y_train.shape}"
        print(f"训练数据维度: {X_train.shape}")
        print(f"测试数据维度: {X_test.shape}")
        
        # 训练模型
        print("\n开始训练Transformer模型...")
        y_pred, y_true, model = train_transformer_model(X_train, y_train, X_test, y_test, progress_callback)
        
        # 验证模型输出
        assert isinstance(model, TimeSeriesTransformer), "模型类型错误"
        assert y_pred.shape == y_true.shape, "预测结果维度与真实值不匹配"
        
        # 反归一化预测结果
        y_pred_actual = price_scaler.inverse_transform(y_pred)
        y_true_actual = price_scaler.inverse_transform(y_true)
        
        # 计算评估指标
        mape = np.mean(np.abs((y_true_actual - y_pred_actual) / y_true_actual)) * 100
        rmse = np.sqrt(np.mean((y_true_actual - y_pred_actual) ** 2))
        mae = np.mean(np.abs(y_true_actual - y_pred_actual))
        
        print("\n模型评估指标:")
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # 生成未来预测
        print(f"\n预测未来{prediction_days}天的股价...")
        
        # 准备预测数据
        last_sequence = X_test[-1:].copy()
        future_predictions = []
        
        # 使用模型进行预测
        model.eval()
        with torch.no_grad():
            for _ in range(prediction_days):
                # 转换为tensor
                last_sequence_tensor = torch.FloatTensor(last_sequence)
                
                # 预测下一个值
                next_pred, _ = model(last_sequence_tensor)
                next_pred = next_pred.numpy()
                
                # 将预测值转换回实际价格
                pred_reshaped = np.array([[next_pred[-1]]], dtype=np.float32)
                actual_price = price_scaler.inverse_transform(pred_reshaped)[0, 0]
                future_predictions.append(float(actual_price))
                
                # 更新序列
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred[-1]
        
        # 生成预测日期
        future_dates = pd.date_range(
            start=end_date + timedelta(days=1),
            periods=prediction_days,
            freq='B'  # 使用工作日频率
        )
        
        print("\n未来30天预测结果:")
        for date, price in zip(future_dates, future_predictions):
            print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
        
        # 绘制预测结果图表
        plt.figure(figsize=(12, 6))
        
        # 绘制历史数据
        plt.plot(y_true_actual, label='实际值', color='blue', alpha=0.7)
        plt.plot(y_pred_actual, label='预测值', color='red', alpha=0.7)
        
        # 添加未来预测
        future_indices = np.arange(len(y_true_actual), len(y_true_actual) + len(future_predictions))
        plt.plot(future_indices, future_predictions, label='未来预测', color='green', linestyle='--')
        
        plt.title(f'{stock_code} 股价预测结果')
        plt.xlabel('时间')
        plt.ylabel('股价')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig('transformer_prediction_test.png')
        plt.close()
        
        print("\n测试完成！预测结果图表已保存为 transformer_prediction_test.png")
        
        # 验证预测结果的合理性
        assert all(price > 0 for price in future_predictions), "预测价格存在负值"
        assert max(future_predictions) < max(y_true_actual) * 2, "预测价格异常偏高"
        assert min(future_predictions) > min(y_true_actual) * 0.5, "预测价格异常偏低"
        
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_transformer_model_optimization() 