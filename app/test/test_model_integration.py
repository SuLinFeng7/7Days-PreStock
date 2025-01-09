import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
from datetime import datetime, timedelta
from data.data_preprocessing import get_stock_data
from models.model_trainer import ModelTrainer

class TestModelIntegration(unittest.TestCase):
    def setUp(self):
        """准备测试数据"""
        # 获取一小段测试数据
        self.stock_id = "600104.SH"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 只用1年数据做测试
        
        print("正在获取测试数据...")
        self.test_data = get_stock_data(
            self.stock_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    def test_model_training(self):
        """测试模型训练和预测"""
        try:
            print("\n开始测试模型训练...")
            trainer = ModelTrainer(self.test_data, progress_bar=None)
            predictions, metrics = trainer.train_all_models()
            
            # 检查预测结果
            self.assertIsNotNone(predictions)
            self.assertIsNotNone(metrics)
            
            # 检查是否包含所有模型的结果
            expected_models = ['LSTM', 'CNN', 'Transformer']
            for model in expected_models:
                self.assertIn(model, predictions)
                self.assertIn(model, metrics)
                
                # 检查预测值
                self.assertEqual(len(predictions[model]), 7)  # 应该有7天的预测
                
                # 检查评估指标
                self.assertIn('MAPE', metrics[model])
                self.assertIn('RMSE', metrics[model])
                self.assertIn('MAE', metrics[model])
                
                print(f"\n{model} 模型评估指标:")
                print(f"MAPE: {metrics[model]['MAPE']:.2f}%")
                print(f"RMSE: {metrics[model]['RMSE']:.2f}")
                print(f"MAE: {metrics[model]['MAE']:.2f}")
                
            print("\n所有模型测试通过！")
            
        except Exception as e:
            self.fail(f"模型训练测试失败: {str(e)}")

def main():
    # 设置更详细的测试输出
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 