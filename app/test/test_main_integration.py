import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from main import StockPredictionApp
import tkinter as tk
from datetime import datetime, timedelta
import time

class TestMainIntegration(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.root = tk.Tk()
        self.app = StockPredictionApp()
    
    def test_full_prediction_process(self):
        """测试完整的预测流程"""
        try:
            print("\n开始测试完整预测流程...")
            
            # 设置测试数据
            self.app.stock_id.set("600104.SH")  # 设置股票代码
            
            # 设置固定的日期范围
            start_date = datetime(2025, 1, 9)
            end_date = datetime(2025, 1, 15)
            
            # 设置训练数据的开始日期（往前推1年）
            train_start_date = start_date - timedelta(days=365)
            
            self.app.start_date.set_date(train_start_date)
            self.app.end_date.set_date(start_date)  # 使用预测开始日期作为训练结束日期
            
            # 设置年限
            self.app.year_var.set("1")  # 设置为1年
            
            # 启动预测
            print(f"开始预测 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的股价...")
            self.app.start_prediction()
            
            # 等待预测完成
            max_wait = 300  # 增加到5分钟
            start_time = time.time()
            last_progress = -1
            
            while time.time() - start_time < max_wait:
                if hasattr(self.app, 'latest_predictions') and self.app.latest_predictions is not None:
                    break
                    
                self.root.update()
                
                # 打印进度，但避免重复打印相同的进度
                if hasattr(self.app, 'progress'):
                    current_progress = self.app.progress['value']
                    if current_progress > last_progress:
                        print(f"预测进度: {current_progress:.1f}%")
                        last_progress = current_progress
                
                time.sleep(0.5)  # 增加等待时间，减少CPU使用
            
            if time.time() - start_time >= max_wait:
                raise TimeoutError("预测超时")
            
            # 验证预测结果
            self.assertIsNotNone(self.app.latest_predictions, "预测结果不应为空")
            
            # 检查是否包含所有模型的预测
            expected_models = ['LSTM', 'CNN', 'Transformer']
            print("\n预测结果:")
            print("日期\t\t", end="")
            for model in expected_models:
                print(f"{model}\t\t", end="")
            print()
            
            # 打印每天的预测结果
            current_date = start_date
            for i in range(7):
                print(f"{current_date.strftime('%Y-%m-%d')}", end="\t")
                for model in expected_models:
                    self.assertIn(model, self.app.latest_predictions)
                    predictions = self.app.latest_predictions[model]
                    self.assertEqual(len(predictions), 7)  # 应该有7天的预测
                    print(f"{predictions[i]:.2f}\t\t", end="")
                print()
                current_date += timedelta(days=1)
            
            print("\n预测流程测试通过！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"预测流程测试失败: {str(e)}")
    
    def test_ui_components(self):
        """测试UI组件的功能"""
        try:
            print("\n开始测试UI组件...")
            
            # 测试股票选择
            self.assertIsNotNone(self.app.stock_id)
            self.assertEqual(self.app.stock_id.get(), "600104.SH")  # 默认股票
            
            # 测试日期选择器
            self.assertIsNotNone(self.app.start_date)
            self.assertIsNotNone(self.app.end_date)
            
            # 测试年限选择
            self.assertIsNotNone(self.app.year_var)
            self.assertEqual(self.app.year_var.get(), "3")  # 默认3年
            
            # 测试进度条
            self.assertIsNotNone(self.app.progress)
            
            # 测试结果显示区域
            self.assertIsNotNone(self.app.result_text)
            
            print("UI组件测试通过！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"UI组件测试失败: {str(e)}")
    
    def test_data_validation(self):
        """测试数据验证功能"""
        try:
            print("\n开始测试数据验证...")
            
            # 测试无效日期范围
            future_date = datetime.now() + timedelta(days=30)
            self.app.end_date.set_date(future_date)
            
            # 验证是否有错误提示
            with self.assertRaises(Exception):
                self.app.validate_inputs()
            
            # 测试无效的股票代码
            original_stock = self.app.stock_id.get()
            self.app.stock_id.set("invalid_code")
            
            # 验证是否有错误提示
            with self.assertRaises(Exception):
                self.app.validate_inputs()
            
            # 恢复正确的值
            self.app.stock_id.set(original_stock)
            self.app.end_date.set_date(datetime.now())
            
            print("数据验证测试通过！")
            
        except Exception as e:
            import traceback
            print(f"\n错误详情:\n{traceback.format_exc()}")
            self.fail(f"数据验证测试失败: {str(e)}")
    
    def tearDown(self):
        """清理测试环境"""
        self.root.destroy()

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 