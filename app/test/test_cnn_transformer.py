import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch
from datetime import datetime, timedelta
from data.data_preprocessing import get_stock_data, prepare_prediction_data
from models.cnn_model import train_cnn_model
from models.transformer_model import train_transformer_model

class TestCNNTransformer(unittest.TestCase):
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

    def test_cnn_model(self):
        """测试CNN模型"""
        try:
            print("\n开始测试CNN模型...")
            
            def progress_callback(epoch, total_epochs):
                if epoch % 10 == 0:
                    print(f"CNN训练进度: {epoch}/{total_epochs}")
            
            # 训练CNN模型
            y_pred, y_true, model = train_cnn_model(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                progress_callback
            )
            
            # 生成未来7天预测
            last_sequence = self.X_test[-1:].copy()
            future_predictions = []
            
            print("\nCNN模型未来7天预测:")
            for day in range(7):
                next_pred = model.predict(last_sequence, verbose=0)
                if isinstance(next_pred, np.ndarray):
                    next_pred = next_pred.item()
                
                pred_reshaped = np.array([[next_pred]], dtype=np.float32)
                actual_price = self.scaler.inverse_transform(pred_reshaped)[0, 0]
                future_predictions.append(actual_price)
                
                print(f"第{day+1}天: {actual_price:.2f}")
                
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred
            
            print("\nCNN模型测试完成！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"CNN模型测试失败: {str(e)}")

    def test_transformer_model(self):
        """测试Transformer模型"""
        try:
            print("\n开始测试Transformer模型...")
            
            def progress_callback(epoch, total_epochs):
                if epoch % 5 == 0:
                    print(f"Transformer训练进度: {epoch}/{total_epochs}")
            
            # 训练Transformer模型
            y_pred, y_true, model = train_transformer_model(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                progress_callback
            )
            
            # 生成未来7天预测
            last_sequence = torch.FloatTensor(self.X_test[-1:])
            future_predictions = []
            
            print("\nTransformer模型未来7天预测:")
            for day in range(7):
                with torch.no_grad():
                    next_pred, _ = model(last_sequence)
                    next_pred = next_pred.numpy()
                
                if isinstance(next_pred, np.ndarray):
                    next_pred = next_pred.item()
                
                pred_reshaped = np.array([[next_pred]], dtype=np.float32)
                actual_price = self.scaler.inverse_transform(pred_reshaped)[0, 0]
                future_predictions.append(actual_price)
                
                print(f"第{day+1}天: {actual_price:.2f}")
                
                # 更新序列
                last_sequence = torch.roll(last_sequence, -1, dims=1)
                last_sequence[0, -1, 0] = torch.tensor(next_pred)
            
            print("\nTransformer模型测试完成！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"Transformer模型测试失败: {str(e)}")

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 