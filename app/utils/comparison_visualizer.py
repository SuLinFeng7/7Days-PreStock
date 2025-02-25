import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 修改导入语句
from app.data.data_preprocessing import get_stock_data
from config.config import TUSHARE_TOKEN

class ComparisonVisualizer:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("预测值VS实际值对比")
        self.window.geometry("800x600")
        
        # 创建文件选择框架
        self.create_file_selector()
        
        # 创建图表显示区域
        self.chart_frame = ttk.Frame(self.window)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
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
                # 分割路径并查找包含股票代码的部分
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
                
                print(f"找到股票代码: {stock_code}")  # 调试信息
                
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
            plt.clf()
            plt.close('all')
            
            # 创建新图表
            plt.figure(figsize=(12, 6))
            
            # 绘制实际值（如果有）
            if not actual_df.empty:
                # 处理不同数据源的列名差异
                close_price = None
                if 'close' in actual_df.columns:  # A股数据
                    close_price = actual_df['close']
                elif 'Close' in actual_df.columns:  # 美股数据
                    close_price = actual_df['Close']
                
                if close_price is not None:
                    plt.plot(actual_df.index, close_price, label='实际值', marker='o')
                else:
                    print(f"警告: 未找到收盘价数据，可用列: {actual_df.columns.tolist()}")
            
            # 绘制预测值
            dates = pd.to_datetime(df['日期'])
            plt.plot(dates, df['LSTM预测值'], label='LSTM预测值', marker='s')
            plt.plot(dates, df['CNN预测值'], label='CNN预测值', marker='^')
            plt.plot(dates, df['Transformer预测值'], label='Transformer预测值', marker='*')
            
            plt.title(f"{stock_code}股票预测值VS实际值对比")
            plt.xlabel("日期")
            plt.ylabel("股价(元)")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # 保存当前的图表信息，供后续保存使用
            self.current_plot_info = {
                'stock_code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'stock_name': self.get_stock_name(stock_code)  # 新增获取股票名称
            }
            
            # 在图表框架中显示图表
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 在成功生成图表后启用保存按钮
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
            
            # 生成文件名
            filename = f"{stock_name}{start_date}至{end_date}实际值与预测值对比图.png"
            save_path = os.path.join(save_dir, filename)
            
            # 保存图表
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"对比图已保存至: {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存对比图失败: {str(e)}")
            import traceback
            print(traceback.format_exc()) 