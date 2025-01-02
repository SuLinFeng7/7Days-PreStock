import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from datetime import timedelta
import pandas as pd
import numpy as np

def create_prediction_chart(predictions, start_date, end_date, master):
    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date)
    num_days = len(date_range)
    
    # 创建图形
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # 为每个模型绘制预测线
    for model_name, pred_values in predictions.items():
        # 确保pred_values是列表
        if not isinstance(pred_values, (list, np.ndarray)):
            pred_values = [pred_values] * num_days
        elif len(pred_values) < num_days:
            # 如果预测值数量少于日期数量，复制最后一个值
            last_value = pred_values[-1]
            pred_values = list(pred_values) + [last_value] * (num_days - len(pred_values))
        elif len(pred_values) > num_days:
            # 如果预测值数量多于日期数量，截取需要的部分
            pred_values = pred_values[:num_days]
        
        # 绘制预测线和点
        line = ax.plot(date_range, pred_values, label=f"{model_name} Prediction", marker='o')
        color = line[0].get_color()
        
        # 添加数据标签
        for x, y in zip(date_range, pred_values):
            ax.annotate(f'{y:.2f}',
                       (x, y),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       color=color,
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.5',
                               fc='white',
                               ec=color,
                               alpha=0.7))
    
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.autofmt_xdate()
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.draw()
    
    return canvas.get_tk_widget()

def create_metrics_table(metrics, master):
    # 创建表格框架
    table_frame = tk.Frame(master, relief='ridge', bd=2)
    
    # 设置列宽
    col_widths = [15, 12, 12, 12]  # 调整每列的宽度
    
    # 创建表头
    headers = ['Model', 'MAPE (%)', 'RMSE', 'MAE']
    for col, (header, width) in enumerate(zip(headers, col_widths)):
        label = tk.Label(
            table_frame,
            text=header,
            font=('Arial', 10, 'bold'),
            width=width,
            relief='ridge',
            bg='#f0f0f0'  # 浅灰色背景
        )
        label.grid(row=0, column=col, sticky='nsew', padx=1, pady=1)
    
    # 找出最佳指标
    best_mape = min(metrics.items(), key=lambda x: x[1]['MAPE'])[1]['MAPE']
    best_rmse = min(metrics.items(), key=lambda x: x[1]['RMSE'])[1]['RMSE']
    best_mae = min(metrics.items(), key=lambda x: x[1]['MAE'])[1]['MAE']
    
    # 添加数据行
    for row, (model_name, metric) in enumerate(metrics.items(), start=1):
        # 模型名称
        tk.Label(
            table_frame,
            text=model_name,
            font=('Arial', 10),
            width=col_widths[0],
            relief='ridge'
        ).grid(row=row, column=0, sticky='nsew', padx=1, pady=1)
        
        # MAPE
        value_label = tk.Label(
            table_frame,
            text=f"{metric['MAPE']:.2f}",
            font=('Arial', 10),
            width=col_widths[1],
            relief='ridge'
        )
        if metric['MAPE'] == best_mape:
            value_label.config(fg='green', font=('Arial', 10, 'bold'))
        value_label.grid(row=row, column=1, sticky='nsew', padx=1, pady=1)
        
        # RMSE
        value_label = tk.Label(
            table_frame,
            text=f"{metric['RMSE']:.2f}",
            font=('Arial', 10),
            width=col_widths[2],
            relief='ridge'
        )
        if metric['RMSE'] == best_rmse:
            value_label.config(fg='green', font=('Arial', 10, 'bold'))
        value_label.grid(row=row, column=2, sticky='nsew', padx=1, pady=1)
        
        # MAE
        value_label = tk.Label(
            table_frame,
            text=f"{metric['MAE']:.2f}",
            font=('Arial', 10),
            width=col_widths[3],
            relief='ridge'
        )
        if metric['MAE'] == best_mae:
            value_label.config(fg='green', font=('Arial', 10, 'bold'))
        value_label.grid(row=row, column=3, sticky='nsew', padx=1, pady=1)
    
    # 配置网格权重，使表格单元格大小一致
    for i in range(4):
        table_frame.grid_columnconfigure(i, weight=1)
    
    return table_frame 