import pandas as pd
import os
from datetime import datetime, timedelta
from openpyxl import load_workbook

class RecordKeeper:
    def __init__(self):
        # 使用当前日期创建文件名
        current_date = datetime.now().strftime('%Y%m%d')
        self.excel_path = f'./record/prediction_records_{current_date}.xlsx'
        self.base_columns = [
            '记录日期',           # 记录日期时间
            '股票代码',            # 股票代码
            '训练数据起始日期',      # 训练数据起始日期
            '训练数据结束日期',        # 训练数据结束日期
            '模型名称',            # 模型名称
            'RMSE', 'MAE', 'MAPE',   # 评估指标
            '训练时长（分钟）'       # 训练时长
        ]
        # 确保record目录存在
        os.makedirs('./record', exist_ok=True)
        self._initialize_excel()
        self.records = []
    
    def _initialize_excel(self):
        """如果Excel文件不存在则创建"""
        if not os.path.exists(self.excel_path):
            df = pd.DataFrame(columns=self.base_columns)
            df.to_excel(self.excel_path, index=False)
            print(f"创建新的记录文件: {self.excel_path}")
    
    def add_record(self, stock_code, predictions, metrics, 
                  train_start_date, train_end_date, train_duration):
        """
        添加预测记录
        """
        record = {
            'stock_code': stock_code,
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'train_duration': train_duration,
            'predictions': predictions,
            'metrics': metrics
        }
        self.records.append(record)
    
    def test_write(self):
        """测试Excel文件写入功能"""
        try:
            print("\n开始测试写入...")
            
            # 创建测试数据
            test_dates = pd.date_range(start=datetime.now().date(), periods=7)  # 使用当前日期开始的7天
            test_predictions = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
            test_metrics = {'RMSE': 0.1, 'MAE': 0.08, 'MAPE': 15.5}
            
            # 添加测试记录
            self.add_record(
                stock_code='TEST001',
                predictions=test_predictions,
                metrics=test_metrics,
                train_start_date=datetime.now().date() - timedelta(days=365),
                train_end_date=datetime.now().date(),
                train_duration=0
            )
            
            # 验证文件是否成功写入
            if os.path.exists(self.excel_path):
                df = pd.read_excel(self.excel_path)
                print("\n测试结果:")
                print(f"文件存在: 是")
                print(f"记录数量: {len(df)}")
                print(f"列数量: {len(df.columns)}")
                print(f"列名: {df.columns.tolist()}")
                if len(df) > 0:
                    print("\n最新记录:")
                    print(df.iloc[-1])
                    return True
            return False
            
        except Exception as e:
            print(f"测试写入时发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False 