import pandas as pd
import os
from datetime import datetime, timedelta
from openpyxl import load_workbook

class RecordKeeper:
    def __init__(self):
        self.excel_path = './record/prediction_records.xlsx'
        self.base_columns = [
            '记录日期',           # 记录日期时间
            '股票代码',            # 股票代码
            '训练数据起始日期',      # 训练数据起始日期
            '训练数据结束日期',        # 训练数据结束日期
            '模型名称',            # 模型名称
            'RMSE', 'MAE', 'MAPE'   # 评估指标
        ]
        self._initialize_excel()
    
    def _initialize_excel(self):
        """如果Excel文件不存在则创建"""
        if not os.path.exists(self.excel_path):
            df = pd.DataFrame(columns=self.base_columns)
            df.to_excel(self.excel_path, index=False)
    
    def add_record(self, stock_code, predictions, metrics, pred_dates, model_name, train_start_date, train_end_date):
        """添加新的预测记录"""
        try:
            print(f"\n开始添加记录...")
            print(f"当前Excel文件路径: {self.excel_path}")
            
            # 读取现有记录
            if os.path.exists(self.excel_path):
                df = pd.read_excel(self.excel_path)
            else:
                df = pd.DataFrame(columns=self.base_columns)
            
            print(f"成功读取现有Excel文件，当前列: {df.columns.tolist()}")
            
            # 准备新记录
            new_record = {
                '记录日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '股票代码': stock_code,
                '训练数据起始日期': train_start_date.strftime('%Y-%m-%d'),
                '训练数据结束日期': train_end_date.strftime('%Y-%m-%d'),
                '模型名称': model_name,
                'RMSE': round(metrics['RMSE'], 3),
                'MAE': round(metrics['MAE'], 3),
                'MAPE': round(metrics['MAPE'], 3)
            }
            
            # 添加预测日期作为列名，预测值作为值
            for i, date in enumerate(pred_dates):
                if i < len(predictions):  # 确保有对应的预测值
                    date_str = date.strftime('%Y-%m-%d')
                    new_record[date_str] = round(predictions[i], 3)
                    print(f"添加预测: {date_str} = {predictions[i]}")
                    # 如果这个日期列不存在，添加到DataFrame中
                    if date_str not in df.columns:
                        df[date_str] = None
                        print(f"新增日期列: {date_str}")
            
            # 添加新记录到DataFrame
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            print(f"新记录已添加，当前记录数: {len(df)}")
            
            # 保存更新后的记录
            df.to_excel(self.excel_path, index=False)
            print(f"成功保存Excel文件")
            print(f"当前所有列: {df.columns.tolist()}")
            print(f"最新一条记录: {new_record}")
            
        except Exception as e:
            print(f"添加记录时发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
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
                pred_dates=test_dates,
                model_name='TEST_MODEL',
                train_start_date=datetime.now().date() - timedelta(days=365),
                train_end_date=datetime.now().date()
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