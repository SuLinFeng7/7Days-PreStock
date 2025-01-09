import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np
from models.transformer_model import Transformer, train_transformer_model
from config.config import TRANSFORMER_PARAMS
from data.data_preprocessing import get_stock_data, prepare_prediction_data
from datetime import datetime, timedelta

class TestTransformerOnly(unittest.TestCase):
    def setUp(self):
        """准备真实股票数据"""
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
            
        except Exception as e:
            print(f"数据准备失败: {str(e)}")
            raise e

    def test_model_components(self):
        """测试模型的各个组件"""
        try:
            print("\n开始测试模型组件...")
            
            # 测试模型初始化
            model = Transformer()
            self.assertIsNotNone(model)
            print("模型初始化成功")
            
            # 测试单个小批次的前向传播
            batch_x = torch.FloatTensor(self.X_train[:4])
            print(f"输入形状: {batch_x.shape}")
            
            outputs, _ = model(batch_x)
            print(f"输出形状: {outputs.shape}")
            print(f"输出样例: {outputs[:3].detach().numpy()}")
            
            # 测试损失计算
            target = torch.FloatTensor(self.y_train[:4])
            criterion = torch.nn.MSELoss()
            loss = criterion(outputs, target)
            print(f"损失值: {loss.item():.6f}")
            
            print("\n模型组件测试通过！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"模型组件测试失败: {str(e)}")

    def test_small_batch_training(self):
        """测试小批量数据的训练"""
        try:
            print("\n开始小批量训练测试...")
            
            # 只使用部分数据进行测试
            train_size = min(64, len(self.X_train))
            test_size = min(16, len(self.X_test))
            
            X_train_small = self.X_train[:train_size]
            y_train_small = self.y_train[:train_size]
            X_test_small = self.X_test[:test_size]
            y_test_small = self.y_test[:test_size]
            
            def progress_callback(epoch, total_epochs):
                if epoch % 2 == 0:  # 每两个epoch打印一次
                    print(f"训练进度: {epoch}/{total_epochs}")
            
            # 使用较少的训练轮数
            TRANSFORMER_PARAMS['epochs'] = 10
            TRANSFORMER_PARAMS['batch_size'] = 16
            
            y_pred, y_true, model = train_transformer_model(
                X_train_small,
                y_train_small,
                X_test_small,
                y_test_small,
                progress_callback
            )
            
            # 反归一化预测结果
            y_pred_actual = self.scaler.inverse_transform(y_pred)
            y_true_actual = self.scaler.inverse_transform(y_true)
            
            print("\n预测结果示例:")
            for i in range(3):
                print(f"预测值: {y_pred_actual[i][0]:.2f}, 真实值: {y_true_actual[i][0]:.2f}")
            
            # 计算评估指标
            mse = np.mean((y_pred_actual - y_true_actual) ** 2)
            mae = np.mean(np.abs(y_pred_actual - y_true_actual))
            mape = np.mean(np.abs((y_pred_actual - y_true_actual) / y_true_actual)) * 100
            
            print(f"\n评估指标:")
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            print("\n小批量训练测试通过！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"小批量训练测试失败: {str(e)}")

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 