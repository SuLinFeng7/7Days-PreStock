import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
from data.data_preprocessing import get_stock_data
from models.model_trainer import ModelTrainer
from utils.visualization import create_prediction_chart, create_metrics_table
from utils.record_keeper import RecordKeeper
import numpy as np
from ttkthemes import ThemedTk  # 新增主题支持
from config.model_versions import MODEL_VERSIONS  # 添加模型版本配置导入

class StockPredictionApp:
    def __init__(self):
        self.root = ThemedTk(theme="arc")  # 使用现代化主题
        self.root.title("智能股票预测系统")
        self.root.geometry("1200x800")  # 增加窗口大小
        
        # 添加默认股票列表
        self.default_stocks = [
            "600104.SH",  # 上汽集团
            "002594.SZ",  # 比亚迪
            "601127.SH",  # 小康股份
            "TSLA"        # 特斯拉
        ]
        
        # 设置整体样式
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#f0f0f0')
        self.style.configure('Custom.TLabel', background='#f0f0f0', font=('Microsoft YaHei UI', 10))
        self.style.configure('Title.TLabel', font=('Microsoft YaHei UI', 16, 'bold'))
        self.style.configure('Custom.TButton', font=('Microsoft YaHei UI', 10))
        
        self.record_keeper = RecordKeeper()
        self.create_widgets()
        
    def create_widgets(self):
        # 主容器
        main_container = ttk.Frame(self.root, style='Custom.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左侧面板 - 输入区域
        left_panel = ttk.Frame(main_container, style='Custom.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # 标题
        title_label = ttk.Label(
            left_panel, 
            text="股票预测配置", 
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))
        
        # 股票代码输入框和下拉框组合
        stock_frame = ttk.Frame(left_panel, style='Custom.TFrame')
        stock_frame.pack(fill=tk.X, pady=10)
        ttk.Label(stock_frame, text="股票代码:", style='Custom.TLabel').pack(side=tk.LEFT)
        
        # 创建组合框
        self.stock_id = ttk.Combobox(
            stock_frame,
            values=self.default_stocks,
            font=('Microsoft YaHei UI', 10),
            width=15
        )
        self.stock_id.set(self.default_stocks[0])  # 设置默认值
        self.stock_id.pack(side=tk.LEFT, padx=10)
        
        # 允许手动输入
        self.stock_id.configure(state='normal')
        
        # 训练年限选择
        year_frame = ttk.Frame(left_panel, style='Custom.TFrame')
        year_frame.pack(fill=tk.X, pady=10)
        ttk.Label(year_frame, text="训练年限:", style='Custom.TLabel').pack(side=tk.LEFT)
        self.year_var = tk.StringVar(value="3")
        year_choices = [str(i) for i in range(1, 11)]
        year_menu = ttk.Combobox(
            year_frame,
            textvariable=self.year_var,
            values=year_choices,
            width=5,
            state="readonly",
            font=('Microsoft YaHei UI', 10)
        )
        year_menu.pack(side=tk.LEFT, padx=10)
        ttk.Label(year_frame, text="年", style='Custom.TLabel').pack(side=tk.LEFT)
        
        # 日期选择区域
        date_frame = ttk.LabelFrame(
            left_panel, 
            text="预测时间范围",
            style='Custom.TFrame'
        )
        date_frame.pack(fill=tk.X, pady=20)
        
        # 开始日期
        start_date_frame = ttk.Frame(date_frame, style='Custom.TFrame')
        start_date_frame.pack(fill=tk.X, pady=10, padx=10)
        ttk.Label(start_date_frame, text="开始日期:", style='Custom.TLabel').pack(side=tk.LEFT)
        self.start_date = DateEntry(
            start_date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            font=('Microsoft YaHei UI', 10)
        )
        self.start_date.pack(side=tk.LEFT, padx=10)
        
        # 结束日期
        end_date_frame = ttk.Frame(date_frame, style='Custom.TFrame')
        end_date_frame.pack(fill=tk.X, pady=10, padx=10)
        ttk.Label(end_date_frame, text="结束日期:", style='Custom.TLabel').pack(side=tk.LEFT)
        
        # 设置默认结束日期为6天后
        default_end_date = datetime.now().date() + timedelta(days=6)
        self.end_date = DateEntry(
            end_date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            font=('Microsoft YaHei UI', 10),
            year=default_end_date.year,
            month=default_end_date.month,
            day=default_end_date.day
        )
        self.end_date.pack(side=tk.LEFT, padx=10)
        
        # 进度条
        progress_frame = ttk.Frame(left_panel, style='Custom.TFrame')
        progress_frame.pack(fill=tk.X, pady=20)
        self.progress = ttk.Progressbar(
            progress_frame,
            length=300,
            mode='determinate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress.pack()
        
        # 开始预测按钮
        button_frame = ttk.Frame(left_panel, style='Custom.TFrame')
        button_frame.pack(fill=tk.X, pady=20)
        start_button = ttk.Button(
            button_frame,
            text="开始预测",
            command=self.start_prediction,
            style='Custom.TButton'
        )
        start_button.pack(pady=10)
        
        # 右侧面板 - 结果显示区域
        right_panel = ttk.Frame(main_container, style='Custom.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 状态信息显示区域
        status_frame = ttk.LabelFrame(
            right_panel,
            text="运行状态",
            style='Custom.TFrame'
        )
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(
            status_frame,
            height=10,
            width=50,
            font=('Microsoft YaHei UI', 10),
            wrap=tk.WORD,
            bg='#ffffff'
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=scrollbar.set)

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
            
        if start == end:
            messagebox.showerror("错误", "开始日期和结束日期不能是同一天")
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
        
        # 获取选择的训练年限
        train_years = int(self.year_var.get())
        self.update_status(f"开始获取股票 {stock_code} 的历史数据...")
        
        try:
            # 修改训练数据的时间范围
            train_end = datetime.now().date()
            train_start = train_end - timedelta(days=365*train_years)
            
            self.update_status(f"正在获取从 {train_start} 到 {train_end} 的历史数据（{train_years}年）...")
            
            # 格式化日期为YYYY-MM-DD格式
            train_start_str = train_start.strftime('%Y-%m-%d')
            train_end_str = train_end.strftime('%Y-%m-%d')
            
            df = get_stock_data(stock_code, train_start_str, train_end_str)
            
            if df.empty:
                self.update_status("错误：无法获取股票数据，请检查股票代码是否正确")
                return
            
            self.update_status("数据获取成功，开始训练模型...")
            
            # 记录训练开始时间
            train_start_time = datetime.now()
            
            # 训练模型
            trainer = ModelTrainer(df, self.progress)
            predictions, metrics = trainer.train_all_models()
            
            # 记录训练结束时间
            train_end_time = datetime.now()
            train_duration = (train_end_time - train_start_time).total_seconds() / 60  # 转换为分钟
            
            self.update_status("模型训练完成，正在生成预测结果...")
            
            # 显示结果
            self.show_results(predictions, metrics, start_date, end_date, train_start, train_end, train_duration)
            
        except Exception as e:
            self.update_status(f"错误：{str(e)}")
        
    def show_results(self, predictions, metrics, start_date, end_date, train_start, train_end, train_duration):
        # 创建新窗口显示结果
        result_window = tk.Toplevel(self.root)
        result_window.title("预测结果分析")
        result_window.geometry("1000x800")
        
        # 设置结果窗口样式
        result_frame = ttk.Frame(result_window, style='Custom.TFrame')
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 标题
        title_label = ttk.Label(
            result_frame,
            text="股票预测结果分析报告",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))
        
        # 创建选项卡
        notebook = ttk.Notebook(result_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 图表选项卡
        chart_tab = ttk.Frame(notebook, style='Custom.TFrame')
        notebook.add(chart_tab, text="预测趋势图")
        
        # 指标选项卡
        metrics_tab = ttk.Frame(notebook, style='Custom.TFrame')
        notebook.add(metrics_tab, text="模型评估指标")
        
        # 在图表选项卡中添加图表
        try:
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            start_date_str = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            historical_data = get_stock_data(self.stock_id.get(), start_date_str, end_date_str)
        except Exception as e:
            self.update_status(f"获取历史数据时发生错误: {str(e)}")
            historical_data = None
            
        chart_widget = create_prediction_chart(
            predictions,
            start_date,
            end_date,
            chart_tab,
            historical_data=historical_data
        )
        chart_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 在指标选项卡中添加评估指标表格
        metrics_frame = create_metrics_table(metrics, metrics_tab)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 保存预测结果到Excel文件
        try:
            # 创建日期列表
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_range.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
            
            # 创建预测数据DataFrame
            prediction_data = {
                '日期': date_range,
                'LSTM预测值': predictions['LSTM'],
                'CNN预测值': predictions['CNN'],
                'Transformer预测值': predictions['Transformer']
            }
            df_predictions = pd.DataFrame(prediction_data)
            
            # 创建评估指标DataFrame，加入版本信息
            metrics_data = []
            for model_name, metric in metrics.items():
                model_info = MODEL_VERSIONS[model_name]
                metrics_data.append({
                    '模型': model_name,
                    '版本': model_info['version'],
                    'MAPE': metric['MAPE'],
                    'RMSE': metric['RMSE'],
                    'MAE': metric['MAE']
                })
            df_metrics = pd.DataFrame(metrics_data)
            
            # 创建模型参数DataFrame
            model_params_data = []
            for model_name, info in MODEL_VERSIONS.items():
                params = info['parameters']
                params_str = ', '.join([f"{k}: {v}" for k, v in params.items()])
                model_params_data.append({
                    '模型': model_name,
                    '版本': info['version'],
                    '参数配置': params_str
                })
            df_model_params = pd.DataFrame(model_params_data)
            
            # 保存到Excel文件
            filename = f"record/prediction_records_{end_date.strftime('%Y%m%d')}.xlsx"
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_predictions.to_excel(writer, sheet_name='预测结果', index=False)
                df_metrics.to_excel(writer, sheet_name='评估指标', index=False)
                df_model_params.to_excel(writer, sheet_name='模型配置', index=False)
            
            self.update_status(f"预测结果已保存到文件: {filename}")
            
        except Exception as e:
            self.update_status(f"保存预测结果时发生错误: {str(e)}")
            
        # 记录预测记录
        self.record_keeper.add_record(
            stock_code=self.stock_id.get(),
            start_date=start_date,
            end_date=end_date,
            predictions=predictions,
            metrics=metrics,
            train_start_date=train_start,
            train_end_date=train_end,
            train_duration=train_duration
        )

    def update_status(self, message, clear=False):
        """更新状态信息到文本区域"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"
        
        if clear:
            self.result_text.delete(1.0, tk.END)
        
        self.result_text.insert(tk.END, formatted_message)
        self.result_text.see(tk.END)
        self.root.update()

    def get_prediction_data(self):
        """获取当前预测设置的数据，用于测试"""
        return {
            'stock_id': self.stock_id.get(),
            'start_date': self.start_date.get_date(),
            'end_date': self.end_date.get_date()
        }

if __name__ == "__main__":
    app = StockPredictionApp()
    app.root.mainloop()
