import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 修改导入语句
from app.data.data_preprocessing import get_stock_data
from config.config import TUSHARE_TOKEN
from app.utils.utils import calculate_metrics  # 添加评估指标计算函数的导入

class ComparisonVisualizer:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("预测值VS实际值对比")
        self.window.geometry("1200x800")  # 增加窗口大小以适应两个图表
        
        # 创建文件选择框架
        self.create_file_selector()
        
        # 创建图表显示区域
        self.charts_container = ttk.Frame(self.window)
        self.charts_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建左右分栏
        self.left_frame = ttk.Frame(self.charts_container)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.charts_container)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def create_file_selector(self):
        """创建文件选择区域"""
        select_frame = ttk.Frame(self.window)
        select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 左侧文件选择区域
        file_frame = ttk.Frame(select_frame)
        file_frame.pack(side=tk.LEFT)
        
        ttk.Label(file_frame, text="预测结果文件:").pack(side=tk.LEFT)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=5)
        
        # 按钮区域
        button_frame = ttk.Frame(select_frame)
        button_frame.pack(side=tk.LEFT, padx=5)
        
        # 浏览按钮
        browse_btn = ttk.Button(
            button_frame,
            text="浏览",
            command=self.browse_file
        )
        browse_btn.pack(side=tk.LEFT, padx=2)
        
        # 生成对比图按钮
        generate_btn = ttk.Button(
            button_frame,
            text="生成对比图",
            command=self.generate_comparison
        )
        generate_btn.pack(side=tk.LEFT, padx=2)
        
        # 保存对比图按钮
        self.save_btn = ttk.Button(
            button_frame,
            text="保存对比图",
            command=self.save_comparison_chart,
            state='disabled'  # 初始状态为禁用
        )
        self.save_btn.pack(side=tk.LEFT, padx=2)
        
    def browse_file(self):
        """浏览文件"""
        record_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "record")
        filename = filedialog.askopenfilename(
            initialdir=record_dir,
            title="选择预测结果文件",
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
        )
        if filename:
            self.file_path_var.set(filename)
            
    def generate_comparison(self):
        """生成对比图"""
        try:
            file_path = self.file_path_var.get()
            if not file_path:
                messagebox.showwarning("警告", "请先选择预测结果文件")
                return
                
            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name="未来预测结果")
            
            # 获取日期范围
            start_date = pd.to_datetime(df['日期'].iloc[0])
            end_date = pd.to_datetime(df['日期'].iloc[-1])
            
            # 从文件路径中提取股票代码
            try:
                path_parts = file_path.replace('\\', '/').split('/')
                for part in path_parts:
                    if '_' in part and (('.SH' in part) or ('.SZ' in part) or 
                                      any(code in part for code in ['TSLA', 'LI', 'NIO', 'XPEV'])):
                        stock_code = part.split('_')[0]
                        break
                else:
                    raise ValueError("未找到有效的股票代码")
                
                if not (stock_code.endswith('.SH') or stock_code.endswith('.SZ') or 
                       stock_code in ['TSLA', 'LI', 'NIO', 'XPEV']):
                    raise ValueError(f"无效的股票代码: {stock_code}")
                
            except Exception as e:
                messagebox.showerror("错误", f"无法从文件路径中获取股票代码: {str(e)}")
                return
            
            # 获取实际数据
            actual_df = get_stock_data(
                stock_code,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # 清除之前的图表
            plt.close('all')
            
            # 创建预测值vs实际值对比图
            fig1 = plt.figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111)
            
            # 绘制实际值（如果有）
            actual_values = None
            if not actual_df.empty:
                close_price = None
                if 'close' in actual_df.columns:  # A股数据
                    close_price = actual_df['close']
                elif 'Close' in actual_df.columns:  # 美股数据
                    close_price = actual_df['Close']
                
                if close_price is not None:
                    ax1.plot(actual_df.index, close_price, label='实际值', marker='o')
                    actual_values = close_price.values
            
            # 绘制预测值
            dates = pd.to_datetime(df['日期'])
            ax1.plot(dates, df['LSTM预测值'], label='LSTM预测值', marker='s')
            ax1.plot(dates, df['CNN预测值'], label='CNN预测值', marker='^')
            ax1.plot(dates, df['Transformer预测值'], label='Transformer预测值', marker='*')
            
            ax1.set_title(f"{stock_code}股票预测值VS实际值对比")
            ax1.set_xlabel("日期")
            ax1.set_ylabel("股价(元)")
            ax1.legend()
            ax1.grid(True)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 计算评估指标
            metrics = {}
            if actual_values is not None:
                for model in ['LSTM', 'CNN', 'Transformer']:
                    pred_values = df[f'{model}预测值'].values
                    mape, rmse, mae = calculate_metrics(actual_values, pred_values)
                    metrics[model] = {
                        'MAPE': mape,
                        'RMSE': rmse,
                        'MAE': mae
                    }
                
                # 创建评估指标对比图
                fig2 = plt.figure(figsize=(8, 6))
                ax2 = fig2.add_subplot(111)
                
                # 准备数据
                models = list(metrics.keys())
                x = np.arange(len(models))
                width = 0.25
                
                # 绘制柱状图
                mape_bars = ax2.bar(x - width, [metrics[m]['MAPE'] for m in models], width, label='MAPE')
                rmse_bars = ax2.bar(x, [metrics[m]['RMSE'] for m in models], width, label='RMSE')
                mae_bars = ax2.bar(x + width, [metrics[m]['MAE'] for m in models], width, label='MAE')
                
                # 添加数值标签
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax2.annotate(f'{height:.2f}',
                                   xy=(rect.get_x() + rect.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                
                autolabel(mape_bars)
                autolabel(rmse_bars)
                autolabel(mae_bars)
                
                ax2.set_title("模型评估指标对比")
                ax2.set_xticks(x)
                ax2.set_xticklabels(models)
                ax2.legend()
                ax2.grid(True)
                
                # 调整布局
                fig2.tight_layout()
            
            # 在左右分栏中显示图表
            for widget in self.left_frame.winfo_children():
                widget.destroy()
            for widget in self.right_frame.winfo_children():
                widget.destroy()
            
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # 显示预测对比图
            canvas1 = FigureCanvasTkAgg(fig1, master=self.left_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 显示评估指标图
            if actual_values is not None:
                canvas2 = FigureCanvasTkAgg(fig2, master=self.right_frame)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 保存当前的图表信息
            self.current_plot_info = {
                'stock_code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'stock_name': self.get_stock_name(stock_code),
                'metrics': metrics if actual_values is not None else None,
                'figures': [fig1, fig2] if actual_values is not None else [fig1]
            }
            
            # 启用保存按钮
            self.save_btn.configure(state='normal')
            
        except Exception as e:
            messagebox.showerror("错误", f"生成对比图失败: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def get_stock_name(self, stock_code):
        """获取股票名称"""
        stock_names = {
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
        return stock_names.get(stock_code, stock_code)

    def save_comparison_chart(self):
        """保存对比图到指定目录"""
        try:
            if not hasattr(self, 'current_plot_info'):
                messagebox.showerror("错误", "请先生成对比图")
                return
            
            # 创建保存目录
            save_dir = os.path.join("record", "real&actual")
            os.makedirs(save_dir, exist_ok=True)
            
            # 获取当前图表信息
            stock_name = self.current_plot_info['stock_name']
            start_date = self.current_plot_info['start_date'].strftime('%Y-%m-%d')
            end_date = self.current_plot_info['end_date'].strftime('%Y-%m-%d')
            
            # 保存预测对比图
            comparison_filename = f"{stock_name}{start_date}至{end_date}实际值与预测值对比图.png"
            comparison_path = os.path.join(save_dir, comparison_filename)
            self.current_plot_info['figures'][0].savefig(comparison_path, dpi=300, bbox_inches='tight')
            
            # 如果有评估指标图，也保存它
            if len(self.current_plot_info['figures']) > 1:
                metrics_filename = f"{stock_name}{start_date}至{end_date}模型评估指标对比图.png"
                metrics_path = os.path.join(save_dir, metrics_filename)
                self.current_plot_info['figures'][1].savefig(metrics_path, dpi=300, bbox_inches='tight')
                
                messagebox.showinfo("成功", f"对比图和评估指标图已保存至:\n{comparison_path}\n{metrics_path}")
            else:
                messagebox.showinfo("成功", f"对比图已保存至:\n{comparison_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存对比图失败: {str(e)}")
            import traceback
            print(traceback.format_exc()) 