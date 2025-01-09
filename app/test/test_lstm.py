import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from data.data_preprocessing import get_stock_data, prepare_prediction_data
from models.lstm_model import train_lstm_model
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TestLSTM(unittest.TestCase):
    def setUp(self):
        """准备测试数据"""
        try:
            print("\n开始获取股票数据...")
            # 获取最近1年的数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # 获取上汽集团的数据作为测试
            stock_code = "600104.SH"
            df = get_stock_data(
                stock_code,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            print(f"获取到 {len(df)} 条股票数据")
            
            # 准备训练数据
            X_train, y_train, X_test, y_test, self.scaler = prepare_prediction_data(df)
            
            # 转换为numpy数组
            self.X_train = np.array(X_train, dtype=np.float32)
            self.y_train = np.array(y_train, dtype=np.float32)
            self.X_test = np.array(X_test, dtype=np.float32)
            self.y_test = np.array(y_test, dtype=np.float32)
            
            print(f"训练数据形状: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
            print(f"测试数据形状: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
            
            # 创建测试输出目录
            self.test_dir = "./test_output/lstm"
            os.makedirs(self.test_dir, exist_ok=True)
            
        except Exception as e:
            print(f"数据准备失败: {str(e)}")
            raise e

    def test_model_training(self):
        """测试LSTM模型训练过程"""
        try:
            print("\n开始测试LSTM模型训练...")
            
            def progress_callback(epoch, total_epochs):
                if epoch % 5 == 0:  # 每5个epoch打印一次进度
                    print(f"训练进度: {epoch}/{total_epochs}")
            
            # 训练模型
            y_pred, y_true, model = train_lstm_model(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                progress_callback
            )
            
            # 反归一化预测结果
            y_pred_actual = self.scaler.inverse_transform(y_pred)
            y_true_actual = self.scaler.inverse_transform(y_true)
            
            # 计算评估指标
            mse = np.mean((y_pred_actual - y_true_actual) ** 2)
            mae = np.mean(np.abs(y_pred_actual - y_true_actual))
            mape = np.mean(np.abs((y_pred_actual - y_true_actual) / y_true_actual)) * 100
            
            print("\n模型评估指标:")
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            # 绘制预测结果对比图
            self._plot_predictions(y_pred_actual, y_true_actual)
            
            # 测试未来预测
            self._test_future_predictions(model)
            
            print("\nLSTM模型测试完成！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"LSTM模型测试失败: {str(e)}")

    def _plot_predictions(self, y_pred, y_true):
        """绘制预测结果对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='真实值', color='blue')
        plt.plot(y_pred, label='预测值', color='red', linestyle='--')
        plt.title('LSTM模型预测结果对比')
        plt.xlabel('时间步')
        plt.ylabel('股票价格')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plt.savefig(os.path.join(self.test_dir, 'lstm_predictions.png'))
        plt.close()

    def _test_future_predictions(self, model):
        """测试未来7天预测"""
        try:
            print("\n测试未来7天预测...")
            
            # 使用最后一个序列进行预测
            last_sequence = self.X_test[-1:].copy()
            future_predictions = []
            
            # 预测未来7天
            for day in range(7):
                next_pred = model.predict(last_sequence, verbose=0)
                
                # 确保预测值是标量
                if isinstance(next_pred, np.ndarray):
                    next_pred = next_pred.item()
                
                # 反归一化预测值
                pred_reshaped = np.array([[next_pred]], dtype=np.float32)
                actual_price = self.scaler.inverse_transform(pred_reshaped)[0, 0]
                future_predictions.append(actual_price)
                
                print(f"第{day+1}天预测价格: {actual_price:.2f}")
                
                # 更新序列用于下一次预测
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred
            
            # 绘制未来预测趋势图
            self._plot_future_predictions(future_predictions)
            
        except Exception as e:
            print(f"未来预测测试失败: {str(e)}")
            raise e

    def _plot_future_predictions(self, predictions):
        """绘制未来预测趋势图"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 8), predictions, marker='o', linestyle='-', linewidth=2)
        plt.title('LSTM模型未来7天预测趋势')
        plt.xlabel('预测天数')
        plt.ylabel('预测价格')
        plt.grid(True)
        
        # 添加数据标签
        for i, price in enumerate(predictions):
            plt.annotate(f'{price:.2f}', 
                        (i+1, price),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        # 保存图片
        plt.savefig(os.path.join(self.test_dir, 'lstm_future_predictions.png'))
        plt.close()

def main():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 运行测试
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 