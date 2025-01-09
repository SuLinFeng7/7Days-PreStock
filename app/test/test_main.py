import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from tkinter import Tk
from main import StockPredictionApp
import pandas as pd
from datetime import datetime, timedelta

class TestMainApp(unittest.TestCase):
    def setUp(self):
        self.root = Tk()
        self.app = StockPredictionApp()
        
    def test_basic_functionality(self):
        """测试基本功能"""
        try:
            # 设置测试数据
            self.app.stock_id.set("600104.SH")  # 设置股票代码
            
            # 设置日期
            today = datetime.now()
            start_date = today - timedelta(days=365*3)  # 3年前
            end_date = today + timedelta(days=6)  # 6天后
            
            self.app.start_date.set_date(start_date)
            self.app.end_date.set_date(end_date)
            
            # 触发预测
            self.app.start_prediction()
            
            print("基本功能测试通过")
            
        except Exception as e:
            self.fail(f"测试失败: {str(e)}")
            
    def test_data_retrieval(self):
        """测试数据获取功能"""
        try:
            stock_id = "600104.SH"
            start_date = "20220101"
            end_date = "20240101"
            
            from data.data_preprocessing import get_stock_data
            
            # 测试数据获取
            self.app.update_status("开始获取测试数据...")
            data = get_stock_data(stock_id, start_date, end_date)
            
            self.assertIsNotNone(data)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            
            print("数据获取测试通过")
            print(f"获取到 {len(data)} 条数据")
            
        except Exception as e:
            self.fail(f"数据获取测试失败: {str(e)}")
    
    def test_interface_elements(self):
        """测试界面元素是否正确初始化"""
        try:
            # 测试股票代码下拉框
            self.assertIsNotNone(self.app.stock_id)
            self.assertEqual(self.app.stock_id.get(), "600104.SH")  # 检查默认值
            
            # 测试日期选择器
            self.assertIsNotNone(self.app.start_date)
            self.assertIsNotNone(self.app.end_date)
            
            # 测试进度条 - 修改属性名
            self.assertIsNotNone(self.app.progress)  # 改为 progress
            
            # 测试其他界面元素
            self.assertIsNotNone(self.app.result_text)  # 测试状态文本框
            self.assertIsNotNone(self.app.year_var)     # 测试年限选择
            
            # 测试默认值
            self.assertEqual(self.app.year_var.get(), "3")  # 默认3年
            
            print("界面元素测试通过")
            
        except Exception as e:
            self.fail(f"界面元素测试失败: {str(e)}")
    
    def tearDown(self):
        self.root.destroy()

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 