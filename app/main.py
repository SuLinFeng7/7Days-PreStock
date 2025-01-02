import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
from data.data_preprocessing import get_stock_data
from models.model_trainer import ModelTrainer
from utils.visualization import create_prediction_chart, create_metrics_table

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("股票预测系统")
        self.root.geometry("600x400")
        
        # 创建界面元素
        self.create_widgets()
        
    def create_widgets(self):
        # 股票代码输入
        tk.Label(self.root, text="股票代码:").pack(pady=5)
        self.stock_id = tk.Entry(self.root)
        self.stock_id.pack(pady=5)
        
        # 日期选择
        date_frame = tk.Frame(self.root)
        date_frame.pack(pady=10)
        
        tk.Label(date_frame, text="开始日期:").grid(row=0, column=0)
        self.start_date = DateEntry(date_frame, width=12, background='darkblue',
                                  foreground='white', borderwidth=2)
        self.start_date.grid(row=0, column=1, padx=5)
        
        tk.Label(date_frame, text="结束日期:").grid(row=0, column=2)
        self.end_date = DateEntry(date_frame, width=12, background='darkblue',
                                foreground='white', borderwidth=2)
        self.end_date.grid(row=0, column=3, padx=5)
        
        # 进度条
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack(pady=20)
        
        # 开始预测按钮
        tk.Button(self.root, text="开始预测", command=self.start_prediction).pack(pady=10)
        
        # 结果显示区域
        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.pack(pady=10)
        
    def validate_dates(self):
        start = self.start_date.get_date()
        end = self.end_date.get_date()
        today = datetime.now().date()
        
        if start < today:
            messagebox.showerror("错误", "开始日期必须从今天开始")
            return False
            
        if end < start:
            messagebox.showerror("错误", "结束日期必须大于开始日期")
            return False
            
        if (end - start).days > 7:
            messagebox.showerror("错误", "预测期间不能超过7天")
            return False
            
        return True
        
    def start_prediction(self):
        if not self.validate_dates():
            return
            
        stock_code = self.stock_id.get()
        start_date = self.start_date.get_date()
        end_date = self.end_date.get_date()
        
        # 获取历史数据
        try:
            # 修改训练数据的时间范围
            train_end = datetime.now().date()
            train_start = train_end - timedelta(days=365*3)  # 获取3年的历史数据
            
            # 格式化日期为YYYY-MM-DD格式
            train_start_str = train_start.strftime('%Y-%m-%d')
            train_end_str = train_end.strftime('%Y-%m-%d')
            
            df = get_stock_data(stock_code, train_start_str, train_end_str)
            
            if df.empty:
                messagebox.showerror("错误", "无法获取股票数据，请检查股票代码是否正确")
                return
            
        except Exception as e:
            messagebox.showerror("错误", f"获取股票数据失败: {str(e)}")
            return
            
        # 训练模型
        trainer = ModelTrainer(df, self.progress)
        predictions, metrics = trainer.train_all_models()
        
        # 显示结果
        self.show_results(predictions, metrics, start_date, end_date)
        
    def show_results(self, predictions, metrics, start_date, end_date):
        # 创建新窗口显示结果
        result_window = tk.Toplevel(self.root)
        result_window.title("预测结果")
        result_window.geometry("800x800")  # 增加窗口高度以适应所有内容
        
        # 创建一个主框架来容纳所有内容
        main_frame = tk.Frame(result_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加标题
        title_label = tk.Label(
            main_frame,
            text="Stock Price Prediction Results",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # 添加表格（放在图表上方）
        metrics_frame = create_metrics_table(metrics, main_frame)
        metrics_frame.pack(pady=(0, 20), fill=tk.X)
        
        # 添加图表
        chart = create_prediction_chart(predictions, start_date, end_date, main_frame)
        chart.pack(pady=(0, 20), fill=tk.BOTH, expand=True)
        
        # 显示最佳预测结果
        best_model = min(metrics.items(), key=lambda x: x[1]['MAPE'])[0]
        best_prediction = predictions[best_model]
        
        result_label = tk.Label(
            main_frame, 
            text=f"Best Prediction ({best_model} Model): ${best_prediction:.2f}",
            font=('Arial', 12, 'bold')
        )
        result_label.pack(pady=(0, 10))

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()
