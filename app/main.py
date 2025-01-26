import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
from data.data_preprocessing import get_stock_data
from models.model_trainer import ModelTrainer
from utils.visualization import create_prediction_chart, create_metrics_table, create_historical_comparison_chart
from utils.record_keeper import RecordKeeper
import numpy as np
from ttkthemes import ThemedTk  # 新增主题支持
from config.model_versions import MODEL_VERSIONS  # 添加模型版本配置导入
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 添加以下字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

class StockPredictionApp:
    def __init__(self):
        self.root = ThemedTk(theme="arc")  # 使用现代化主题
        self.root.title("智能股票预测系统")
        self.root.geometry("1200x800")  # 增加窗口大小
        
        # 添加默认股票列表
        self.default_stocks = [
            "600104.SH",  # 上汽集团
            "002594.SZ",  # 比亚迪
            "601127.SH",  # 赛力斯
            "600006.SH",  # 长城汽车
            "601633.SH",  # 广汽集团 
            "300750.SZ",  # 宁德时代
            "TSLA",        # 特斯拉
            "LI",          # 理想汽车
            "NIO",         # 蔚来汽车
            "XPEV"         # 小鹏汽车
        ]
        
        # 设置整体样式
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#f0f0f0')
        self.style.configure('Custom.TLabel', background='#f0f0f0', font=('Microsoft YaHei UI', 10))
        self.style.configure('Title.TLabel', font=('Microsoft YaHei UI', 16, 'bold'))
        self.style.configure('Custom.TButton', font=('Microsoft YaHei UI', 10))
        
        # 设置默认训练年限
        self.default_train_years = 10
        # 设置默认预测天数
        self.default_prediction_days = 30
        
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
        self.year_var = tk.StringVar(value=str(self.default_train_years))
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
            date_pattern='yyyy-mm-dd'
        )
        self.start_date.pack(side=tk.LEFT, padx=10)
        
        # 结束日期
        end_date_frame = ttk.Frame(date_frame, style='Custom.TFrame')
        end_date_frame.pack(fill=tk.X, pady=10, padx=10)
        ttk.Label(end_date_frame, text="结束日期:", style='Custom.TLabel').pack(side=tk.LEFT)
        
        # 设置默认结束日期为30天后
        default_end_date = datetime.now() + timedelta(days=self.default_prediction_days)
        self.end_date = DateEntry(
            end_date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
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
        """验证日期选择是否合法"""
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
            
        if (end - start).days > self.default_prediction_days:
            messagebox.showerror("错误", f"预测期间不能超过{self.default_prediction_days}天")
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
            predictions, historical_predictions, metrics = trainer.train_all_models()
            
            # 记录训练结束时间
            train_end_time = datetime.now()
            train_duration = (train_end_time - train_start_time).total_seconds() / 60  # 转换为分钟
            
            self.update_status("模型训练完成，正在生成预测结果...")
            
            # 显示结果
            self.show_results(
                predictions, 
                historical_predictions,  # 添加历史预测数据
                metrics, 
                start_date, 
                end_date, 
                train_start, 
                train_end, 
                train_duration
            )
            
        except Exception as e:
            self.update_status(f"错误：{str(e)}")
        
    def show_results(self, predictions, historical_predictions, metrics, start_date, end_date, train_start, train_end, train_duration):
        """显示预测结果"""
        try:
            # 获取股票代码和日期信息用于文件命名
            stock_code = self.stock_id.get()
            date_str = end_date.strftime('%Y%m%d')
            
            # 创建记录目录结构
            record_dir = f"record/{stock_code}_{date_str}"
            os.makedirs(record_dir, exist_ok=True)
            os.makedirs(f"{record_dir}/images", exist_ok=True)
            
            # 设置中文字体路径
            font_path = 'app/utils/SimHei.ttf'
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 保存历史预测对比图
            hist_fig = plt.Figure(figsize=(12, 6))
            hist_ax = hist_fig.add_subplot(111)
            
            for model_name, data in historical_predictions.items():
                if '_historical' in model_name:
                    model_display_name = model_name.replace('_historical', '')
                    hist_ax.plot(data['predicted'], label=f"{model_display_name}_预测", alpha=0.7)
            
            hist_ax.plot(next(iter(historical_predictions.values()))['actual'], 
                        label="实际值", linewidth=2, color='black')
            
            hist_ax.set_title("最近两年预测对比", fontproperties='SimHei', fontsize=12)
            hist_ax.set_xlabel("时间", fontproperties='SimHei')
            hist_ax.set_ylabel("股价", fontproperties='SimHei')
            hist_ax.legend(prop={'family': 'SimHei'})
            hist_ax.grid(True)
            
            # 保存历史预测图
            hist_fig.savefig(f"{record_dir}/images/historical_comparison.png", 
                            bbox_inches='tight', dpi=300)
            
            # 2. 保存未来预测图
            future_fig = plt.Figure(figsize=(12, 6))
            future_ax = future_fig.add_subplot(111)
            
            # 修改未来预测数据的保存部分
            date_range = pd.date_range(start=start_date, periods=len(next(iter(predictions.values()))))
            
            # 绘制未来预测图
            for model_name, pred_values in predictions.items():
                model_display_name = model_name.replace('_future', '')
                future_ax.plot(date_range, pred_values, 
                              label=f'{model_display_name}预测', 
                              marker='o')
            
            future_ax.set_title("未来预测趋势", fontproperties='SimHei', fontsize=12)
            future_ax.set_xlabel("预测日期", fontproperties='SimHei')
            future_ax.set_ylabel("预测价格", fontproperties='SimHei')
            future_ax.legend(prop={'family': 'SimHei'})
            future_ax.grid(True)
            
            # 设置x轴日期格式
            future_ax.tick_params(axis='x', rotation=45)
            future_fig.autofmt_xdate()  # 自动调整日期标签
            
            # 保存未来预测图
            future_fig.savefig(f"{record_dir}/images/future_prediction.png", 
                              bbox_inches='tight', dpi=300)
            
            # 创建预测数据DataFrame用于保存到Excel
            prediction_data = {
                '日期': date_range.strftime('%Y-%m-%d')
            }
            
            # 添加每个模型的预测值到DataFrame
            for model_name, pred_values in predictions.items():
                model_display_name = model_name.replace('_future', '')
                prediction_data[f'{model_display_name}预测值'] = [round(x, 2) for x in pred_values]
            
            df_predictions = pd.DataFrame(prediction_data)
            
            # 3. 保存评估指标图
            metrics_fig = plt.Figure(figsize=(12, 6))
            metrics_ax = metrics_fig.add_subplot(111)
            
            # 准备评估指标数据
            model_names = []
            mape_values = []
            rmse_values = []
            mae_values = []
            training_times = []  # 添加训练时长列表
            
            for model_name, metric in metrics.items():
                if '_historical' in model_name:
                    model_names.append(model_name.replace('_historical', ''))
                    mape_values.append(metric['MAPE'])
                    rmse_values.append(metric['RMSE'])
                    mae_values.append(metric['MAE'])
                    if 'training_time' in metric:  # 添加训练时长
                        training_times.append(metric['training_time'])
            
            x = np.arange(len(model_names))
            width = 0.25
            
            metrics_ax.bar(x - width, mape_values, width, label='MAPE (%)')
            metrics_ax.bar(x, rmse_values, width, label='RMSE')
            metrics_ax.bar(x + width, mae_values, width, label='MAE')
            
            metrics_ax.set_title("模型评估指标对比", fontproperties='SimHei', fontsize=12)
            metrics_ax.set_xticks(x)
            metrics_ax.set_xticklabels(model_names, fontproperties='SimHei')
            metrics_ax.legend(prop={'family': 'SimHei'})
            
            # 保存评估指标图
            metrics_fig.savefig(f"{record_dir}/images/metrics_comparison.png", 
                              bbox_inches='tight', dpi=300)
            
            # 4. 保存Excel数据
            # 创建评估指标DataFrame
            metrics_data = []
            for model_name, metric in metrics.items():
                if '_historical' in model_name:
                    model_info = MODEL_VERSIONS[model_name.replace('_historical', '')]
                    training_time = metric.get('training_time', 0)  # 获取训练时长，如果不存在则为0
                    
                    # 计算训练集和对照集的大小
                    total_days = (train_end - train_start).days
                    train_size = int(total_days * 0.8)  # 80%用于训练
                    validation_size = total_days - train_size  # 20%用于验证
                    
                    metrics_data.append({
                        '模型': model_name.replace('_historical', ''),
                        '版本': model_info['version'],
                        'MAPE (%)': f"{metric['MAPE']:.2f}",
                        'RMSE': f"{metric['RMSE']:.2f}",
                        'MAE': f"{metric['MAE']:.2f}",
                        '训练开始日期': train_start.strftime('%Y-%m-%d'),
                        '训练结束日期': train_end.strftime('%Y-%m-%d'),
                        '训练时长(分钟)': f"{training_time:.2f}",
                        '总数据天数': total_days,
                        '训练集天数': train_size,
                        '对照集天数': validation_size,
                        '训练集比例': '80%',
                        '对照集比例': '20%'
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
                    '参数配置': params_str,
                    '预测开始日期': start_date.strftime('%Y-%m-%d'),
                    '预测结束日期': end_date.strftime('%Y-%m-%d'),
                    '预测天数': (end_date - start_date).days + 1
                })
            df_model_params = pd.DataFrame(model_params_data)
            
            # 修改历史预测数据的保存部分
            historical_data = {}
            dates = pd.date_range(
                end=train_end,
                periods=len(next(iter(historical_predictions.values()))['actual']),
                freq='B'  # 使用工作日频率
            )

            # 添加日期列
            historical_data['日期'] = dates

            # 添加每个模型的预测数据
            for model_name, data in historical_predictions.items():
                if '_historical' in model_name:
                    model_display_name = model_name.replace('_historical', '')
                    historical_data[f'{model_display_name}_历史预测'] = data['predicted']

            # 添加实际值
            historical_data['实际值'] = next(iter(historical_predictions.values()))['actual']

            # 创建DataFrame并设置日期索引
            df_historical = pd.DataFrame(historical_data)
            df_historical.set_index('日期', inplace=True)
            
            # 保存Excel文件
            excel_file = f"{record_dir}/prediction_results.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 保存未来预测结果
                df_predictions.to_excel(
                    writer, 
                    sheet_name='未来预测结果',
                    index=False,
                    float_format='%.2f'  # 保留两位小数
                )
                
                # 保存历史预测结果
                df_historical.to_excel(
                    writer, 
                    sheet_name='历史预测结果',
                    float_format='%.2f'  # 保留两位小数
                )
                
                # 保存评估指标
                df_metrics.to_excel(
                    writer, 
                    sheet_name='评估指标',
                    index=False
                )
                
                # 保存模型配置
                df_model_params.to_excel(
                    writer, 
                    sheet_name='模型配置',
                    index=False
                )
            
            self.update_status(f"预测结果已保存到目录: {record_dir}")
            
            # 显示结果窗口（其余显示代码保持不变）
            result_window = tk.Toplevel(self.root)
            result_window.title("预测结果分析")
            result_window.geometry("1200x800")
            
            # 创建选项卡
            notebook = ttk.Notebook(result_window)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # 历史预测对比选项卡
            historical_tab = ttk.Frame(notebook)
            notebook.add(historical_tab, text="历史预测对比")
            
            # 使用 matplotlib 创建历史预测对比图表
            fig = plt.Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            # 绘制每个模型的预测结果
            for model_name, data in historical_predictions.items():
                if '_historical' in model_name:
                    model_display_name = model_name.replace('_historical', '')
                    ax.plot(data['predicted'], label=f"{model_display_name}_预测", alpha=0.7)
            
            # 绘制实际值
            ax.plot(next(iter(historical_predictions.values()))['actual'], 
                    label="实际值", linewidth=2, color='black')
            
            ax.set_title("最近两年预测对比", fontproperties='SimHei', fontsize=12)
            ax.set_xlabel("时间", fontproperties='SimHei')
            ax.set_ylabel("股价", fontproperties='SimHei')
            ax.legend(prop={'family': 'SimHei'})
            ax.grid(True)
            
            # 将图表嵌入到 Tkinter 界面
            canvas = FigureCanvasTkAgg(fig, master=historical_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 评估指标选项卡
            metrics_tab = ttk.Frame(notebook)
            notebook.add(metrics_tab, text="模型评估")
            
            # 创建评估指标表格
            metrics_frame = ttk.Frame(metrics_tab)
            metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 添加评估指标表格
            columns = ('模型', 'MAPE (%)', 'RMSE', 'MAE')
            tree = ttk.Treeview(metrics_frame, columns=columns, show='headings')
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # 添加历史预测评估结果
            for model_name, metric in metrics.items():
                if '_historical' in model_name:
                    model_display_name = model_name.replace('_historical', '')
                    tree.insert('', 'end', values=(
                        model_display_name,
                        f"{metric['MAPE']:.2f}",
                        f"{metric['RMSE']:.2f}",
                        f"{metric['MAE']:.2f}"
                    ))
            
            tree.pack(fill=tk.BOTH, expand=True)
            
            # 未来预测选项卡
            future_tab = ttk.Frame(notebook)
            notebook.add(future_tab, text="未来预测")
            
            # 使用 matplotlib 创建未来预测图表
            future_fig = plt.Figure(figsize=(12, 6))
            future_ax = future_fig.add_subplot(111)
            
            # 绘制每个模型的未来预测
            for model_name, pred_values in predictions.items():
                future_ax.plot(pred_values, label=model_name, marker='o')
            
            future_ax.set_title("未来预测趋势", fontproperties='SimHei', fontsize=12)
            future_ax.set_xlabel("预测天数", fontproperties='SimHei')
            future_ax.set_ylabel("预测价格", fontproperties='SimHei')
            future_ax.legend(prop={'family': 'SimHei'})
            future_ax.grid(True)
            
            # 将图表嵌入到 Tkinter 界面
            future_canvas = FigureCanvasTkAgg(future_fig, master=future_tab)
            future_canvas.draw()
            future_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 记录预测记录
            self.record_keeper.add_record(
                stock_code=self.stock_id.get(),
                predictions=predictions,
                metrics=metrics,
                train_start_date=train_start,
                train_end_date=train_end,
                train_duration=train_duration
            )
            
        except Exception as e:
            self.update_status(f"保存预测结果时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

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
