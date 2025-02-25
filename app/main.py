import sys
import os
import shutil
from PIL import Image, ImageTk

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
from app.data.data_preprocessing import get_stock_data
from models.model_trainer import ModelTrainer
from utils.visualization import create_prediction_chart, create_metrics_table, create_historical_comparison_chart
# from utils.record_keeper import RecordKeeper
import numpy as np
from ttkthemes import ThemedTk  # 新增主题支持
from config.model_versions import MODEL_VERSIONS  # 添加模型版本配置导入
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from app.utils.comparison_visualizer import ComparisonVisualizer

# 添加以下字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

class StockPredictionApp:
    def __init__(self):
        self.root = ThemedTk(theme="arc")
        self.root.title("智能股票预测系统")
        self.root.geometry("1080x720")
        
        # 创建并配置全局样式
        self.style = ttk.Style()
        self.style.configure(
            'Custom.TButton',
            padding=5,
            font=('微软雅黑', 10)
        )
        
        # 配置标签框架样式
        self.style.configure(
            'Custom.TLabelframe',
            background='white',
            relief='solid'
        )
        self.style.configure(
            'Custom.TLabelframe.Label',
            font=('微软雅黑', 10, 'bold'),
            background='white'
        )
        
        # 配置其他基础样式
        self.style.configure('Custom.TFrame', background='#f0f0f0')
        self.style.configure('Custom.TLabel', font=('微软雅黑', 10))
        self.style.configure('Title.TLabel', font=('微软雅黑', 16, 'bold'))
        
        # 初始化其他属性
        self.stock_names = {
            "600104.SH": "上汽集团",
            "002594.SZ": "比亚迪",
            "601127.SH": "赛力斯",
            "600006.SH": "长城汽车", 
            "601633.SH": "广汽集团",
            "300750.SZ": "宁德时代",
            "TSLA": "特斯拉",
            "LI": "理想汽车",
            "NIO": "蔚来汽车",
            "XPEV": "小鹏汽车"
        }
        
        self.default_stocks = [f"{code} - {name}" for code, name in self.stock_names.items()]
        self.default_train_years = 10
        self.default_prediction_days = 30
        self.default_compare_start_year = 2020
        self.default_compare_end_year = datetime.now().year
        
        # 添加进度条属性
        self.progress = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # 创建主容器，使用网格布局
        main_container = ttk.Frame(self.root, style='Custom.TFrame', padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 创建左右分栏
        left_panel = ttk.Frame(main_container, style='Custom.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        right_panel = ttk.Frame(main_container, style='Custom.TFrame')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # === 左侧面板内容 ===
        # 预测配置区域
        config_frame = ttk.LabelFrame(
            left_panel,
            text="预测配置",
            style='Custom.TLabelframe',
            padding="10"
        )
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 股票选择
        stock_frame = ttk.Frame(config_frame)
        stock_frame.pack(fill=tk.X, pady=5)
        ttk.Label(stock_frame, text="股票代码:", style='Custom.TLabel').pack(side=tk.LEFT)
        self.stock_id = ttk.Combobox(
            stock_frame,
            values=self.default_stocks,
            width=25,
            state='readonly'
        )
        self.stock_id.set(self.default_stocks[0])
        self.stock_id.pack(side=tk.LEFT, padx=5)
        
        # 训练年限选择（用于预测）
        year_frame = ttk.Frame(config_frame)
        year_frame.pack(fill=tk.X, pady=5)
        ttk.Label(year_frame, text="训练年限:", style='Custom.TLabel').pack(side=tk.LEFT)
        self.train_year_var = ttk.Combobox(
            year_frame,
            values=[str(i) for i in range(1, 11)],
            width=5,
            state='readonly'
        )
        self.train_year_var.set(str(self.default_train_years))
        self.train_year_var.pack(side=tk.LEFT, padx=5)
        
        # 添加预测日期选择
        date_frame = ttk.Frame(config_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        # 开始日期选择
        ttk.Label(date_frame, text="预测开始日期:", style='Custom.TLabel').pack(side=tk.LEFT)
        self.start_date = DateEntry(
            date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd'
        )
        self.start_date.pack(side=tk.LEFT, padx=5)
        
        # 结束日期选择
        ttk.Label(date_frame, text="预测结束日期:", style='Custom.TLabel').pack(side=tk.LEFT, padx=(10, 0))
        self.end_date = DateEntry(
            date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd'
        )
        self.end_date.pack(side=tk.LEFT, padx=5)
        
        # 设置默认日期
        default_start = datetime.now()
        default_end = default_start + timedelta(days=self.default_prediction_days)
        self.start_date.set_date(default_start)
        self.end_date.set_date(default_end)
        
        # 添加进度条
        progress_frame = ttk.Frame(config_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=200
        )
        self.progress.pack(fill=tk.X)
        
        # 在config_frame中添加预测按钮
        predict_button_frame = ttk.Frame(config_frame)
        predict_button_frame.pack(fill=tk.X, pady=10)
        
        # 添加"开始预测"按钮
        start_button = ttk.Button(
            predict_button_frame,
            text="开始预测",
            command=self.start_prediction,
            style='Custom.TButton'
        )
        start_button.pack(side=tk.LEFT, padx=5)
        
        # 添加"全部预测"按钮
        all_predict_button = ttk.Button(
            predict_button_frame,
            text="全部预测",
            command=self.predict_all_stocks,
            style='Custom.TButton'
        )
        all_predict_button.pack(side=tk.LEFT)
        
        # 在predict_button_frame中添加新按钮
        self.compare_pred_actual_button = ttk.Button(
            predict_button_frame,
            text="预测值VS实际值",
            command=self.show_comparison_window,
            style='Custom.TButton'
        )
        self.compare_pred_actual_button.pack(side=tk.LEFT, padx=5)
        
        # === 趋势对比区域 ===
        compare_frame = ttk.LabelFrame(
            left_panel,
            text="股票趋势对比",
            style='Custom.TLabelframe',
            padding="10"
        )
        compare_frame.pack(fill=tk.BOTH, expand=True)
        
        # 股票多选列表
        list_frame = ttk.Frame(compare_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(list_frame, text="选择要对比的股票:", style='Custom.TLabel').pack(anchor=tk.W)
        
        # 列表框和滚动条
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.stock_listbox = tk.Listbox(
            list_container,
            selectmode=tk.MULTIPLE,
            font=('微软雅黑', 10),
            exportselection=False
        )
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.stock_listbox.yview)
        
        self.stock_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stock_listbox.configure(yscrollcommand=scrollbar.set)
        
        # 添加股票到列表
        for stock_code, stock_name in self.stock_names.items():
            self.stock_listbox.insert(tk.END, f"{stock_code} - {stock_name}")
        
        # 趋势对比的年份选择（用于对比图）
        year_select_frame = ttk.Frame(compare_frame)
        year_select_frame.pack(fill=tk.X, pady=5)
        
        # 起始年份
        ttk.Label(year_select_frame, text="起始年份:", style='Custom.TLabel').pack(side=tk.LEFT)
        self.compare_start_year = ttk.Spinbox(
            year_select_frame,
            from_=2010,
            to=datetime.now().year,
            width=6,
            state='readonly'
        )
        self.compare_start_year.pack(side=tk.LEFT, padx=5)
        self.compare_start_year.set(str(self.default_compare_start_year))
        
        ttk.Label(year_select_frame, text="结束年份:", style='Custom.TLabel').pack(side=tk.LEFT, padx=(10, 0))
        self.compare_end_year = ttk.Spinbox(
            year_select_frame,
            from_=2010,
            to=datetime.now().year + 1,
            width=6,
            state='readonly'
        )
        self.compare_end_year.pack(side=tk.LEFT, padx=5)
        self.compare_end_year.set(str(self.default_compare_end_year))
        
        # 按钮区域
        button_frame = ttk.Frame(compare_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.compare_button = ttk.Button(
            button_frame,
            text="生成趋势对比图",
            command=self.generate_comparison_chart,
            style='Custom.TButton'
        )
        self.compare_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.save_button = ttk.Button(
            button_frame,
            text="保存对比图",
            command=self.save_comparison_chart,
            state=tk.DISABLED,
            style='Custom.TButton'
        )
        self.save_button.pack(side=tk.LEFT)
        
        # === 右侧面板内容 ===
        # 状态显示区域
        status_frame = ttk.LabelFrame(
            right_panel,
            text="运行状态",
            style='Custom.TLabelframe',
            padding="10"
        )
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        # 状态文本框
        self.result_text = tk.Text(
            status_frame,
            height=20,
            width=50,
            font=('微软雅黑', 10)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def on_stock_select(self, event=None):
        """处理股票选择事件"""
        selected = self.stock_id.get()
        if selected:
            # 如果是手动输入的代码，检查是否在映射中
            if selected in self.stock_names:
                self.stock_id.set(f"{selected} - {self.stock_names[selected]}")
            elif " - " in selected:
                # 已经是格式化的显示，不需要处理
                pass
            else:
                # 未知的股票代码，保持原样
                pass

    def validate_dates(self):
        """验证日期选择是否有效"""
        try:
            start = self.start_date.get_date()
            end = self.end_date.get_date()
            
            if start >= end:
                messagebox.showwarning("警告", "开始日期必须早于结束日期")
                return False
            
            return True
        except Exception as e:
            messagebox.showerror("错误", f"日期验证失败: {str(e)}")
            return False
        
    def start_prediction(self):
        """开始预测"""
        try:
            if not self.validate_dates():
                return
            
            # 获取当前选择的股票代码
            stock_display = self.stock_id.get()
            stock_code = stock_display.split(" - ")[0]
            
            # 获取训练年限
            train_years = int(self.train_year_var.get())
            
            # 获取预测日期范围
            start_date = self.start_date.get_date()
            end_date = self.end_date.get_date()
            
            self.update_status(f"开始获取股票 {stock_code} 的历史数据...")
            
            try:
                # 修改训练数据的时间范围
                train_end = datetime.now().date()
                train_start = train_end - timedelta(days=365*train_years)
                
                self.update_status(f"正在获取从 {train_start} 到 {train_end} 的历史数据（{train_years}年）...")
                
                # 格式化日期为YYYY-MM-DD格式
                train_start_str = train_start.strftime('%Y-%m-%d')
                train_end_str = train_end.strftime('%Y-%m-%d')
                
                self.data = get_stock_data(stock_code, train_start_str, train_end_str)
                
                if self.data.empty:
                    self.update_status("错误：无法获取股票数据，请检查股票代码是否正确")
                    return
                
                self.update_status("数据获取成功，开始训练模型...")
                
                # 记录训练开始时间
                train_start_time = datetime.now()
                
                # 训练模型
                trainer = ModelTrainer(self.data, self.progress)
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
            
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")
        
    def show_results(self, predictions, historical_predictions, metrics, start_date, end_date, train_start, train_end, train_duration):
        """显示预测结果"""
        try:
            # 从显示文本中提取股票代码
            stock_display = self.stock_id.get()
            stock_code = stock_display.split(" - ")[0] if " - " in stock_display else stock_display
            
            # 获取股票名称
            stock_name = self.stock_names.get(stock_code, "")
            
            # 创建记录目录结构
            record_dir = f"record/{stock_code}_{end_date.strftime('%Y%m%d')}"
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
            
            hist_ax.set_title(f"{stock_code} {stock_name} - 最近两年预测对比", 
                            fontproperties='SimHei', fontsize=12)
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
            
            future_ax.set_title(f"{stock_code} {stock_name} - 未来预测趋势", 
                              fontproperties='SimHei', fontsize=12)
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
                    
                    # 计算实际的数据集大小
                    total_days = len(self.data)  # 使用实际的数据长度
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

    def predict_all_stocks(self):
        """预测默认股票列表中的所有股票"""
        for stock_display in self.default_stocks:
            stock_code = stock_display.split(" - ")[0]  # 提取股票代码
            self.stock_id.set(stock_display)  # 设置当前股票
            self.update_status(f"开始预测股票: {stock_code}")
            
            # 设置默认的日期范围
            start_date = datetime.now().date()
            end_date = start_date + timedelta(days=self.default_prediction_days)
            self.start_date.set_date(start_date)
            self.end_date.set_date(end_date)
            
            # 调用现有的预测方法
            self.start_prediction()
            self.update_status(f"完成预测股票: {stock_code}")

    def generate_comparison_chart(self):
        """生成股票趋势对比图"""
        try:
            # 获取选中的股票
            selected_indices = self.stock_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("警告", "请至少选择一支股票")
                return
            
            # 获取对比图的年份范围
            start_year = int(self.compare_start_year.get())
            end_year = int(self.compare_end_year.get())
            
            if start_year >= end_year:
                messagebox.showwarning("警告", "起始年份必须小于结束年份")
                return
            
            # 设置日期范围
            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)
            
            # 清除之前的图表
            plt.clf()
            plt.close('all')  # 确保关闭所有图表
            
            # 创建新图表
            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
            
            # 获取并绘制每支股票的数据
            selected_stocks = []
            for idx in selected_indices:
                stock_info = self.stock_listbox.get(idx).split(" - ")
                stock_code = stock_info[0]
                stock_name = stock_info[1]
                selected_stocks.append(stock_name)
                
                # 获取股票数据
                df = get_stock_data(
                    stock_code,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # 确保数据在指定的日期范围内
                df = df[start_date:end_date]
                
                # 绘制股价走势
                ax.plot(df.index, df['close'], label=f"{stock_name}({stock_code})")
            
            # 设置x轴范围
            ax.set_xlim([start_date, end_date])
            
            # 设置图表标题和标签
            plt.title(f"{start_year}年至{end_year}年新能源汽车股票波动趋势对比")
            plt.xlabel("日期")
            plt.ylabel("股价(元)")
            plt.legend()
            plt.grid(True)
            
            # 调整日期显示
            fig.autofmt_xdate()
            
            # 保存图表
            self.temp_chart_path = "temp_comparison_chart.png"
            plt.savefig(self.temp_chart_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            
            # 更新文件名
            stock_names = '-'.join(selected_stocks)
            self.save_filename = f"{start_year}年至{end_year}年{stock_names}股价波动趋势对比图"
            
            # 显示图表
            self.show_chart_in_window()
            
            # 启用保存按钮
            self.save_button.configure(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("错误", f"生成对比图失败: {str(e)}")
            print(f"错误详情: {str(e)}")
        
    def show_chart_in_window(self):
        """在新窗口中显示图表"""
        chart_window = tk.Toplevel(self.root)
        chart_window.title("股票趋势对比图")
        
        # 设置窗口大小为屏幕的80%
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # 加载图片并调整大小
        img = Image.open(self.temp_chart_path)
        # 计算调整后的尺寸，保持宽高比
        img_width, img_height = img.size
        ratio = min(window_width/img_width, window_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # 创建标签显示图片
        label = tk.Label(chart_window, image=photo)
        label.image = photo  # 保持引用
        label.pack(padx=10, pady=10)
        
        # 设置窗口大小和位置
        x = (screen_width - new_width) // 2
        y = (screen_height - new_height) // 2
        chart_window.geometry(f"{new_width + 20}x{new_height + 20}+{x}+{y}")
        
        # 添加关闭按钮
        close_button = ttk.Button(
            chart_window,
            text="关闭",
            command=chart_window.destroy,
            style='Custom.TButton'
        )
        close_button.pack(pady=5)

    def save_comparison_chart(self):
        """保存对比图到record目录"""
        try:
            # 创建record/compare目录（如果不存在）
            save_dir = os.path.join("record", "compare")
            os.makedirs(save_dir, exist_ok=True)
            
            # 使用格式化的文件名
            filename = f"{self.save_filename}.png"
            save_path = os.path.join(save_dir, filename)
            
            # 复制临时文件到目标位置
            shutil.copy2(self.temp_chart_path, save_path)
            
            messagebox.showinfo("成功", f"对比图已保存至: {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存对比图失败: {str(e)}")

    def show_comparison_window(self):
        """显示预测值与实际值对比窗口"""
        ComparisonVisualizer(self.root)

if __name__ == "__main__":
    app = StockPredictionApp()
    app.root.mainloop()
