import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

def create_prediction_chart(predictions, start_date, end_date, parent_frame, historical_data=None):
    """
    创建预测图表，包含历史数据和预测数据
    """
    # 生成预测日期列表
    pred_dates = pd.date_range(start=start_date, end=end_date)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制历史数据（如果有）
    if historical_data is not None:
        # 获取最近7天的历史数据
        end_date_hist = pd.Timestamp(datetime.now().date())
        start_date_hist = end_date_hist - timedelta(days=7)
        
        # 确保索引是datetime类型
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            historical_data.index = pd.to_datetime(historical_data.index)
        
        # 使用datetime64类型进行过滤
        mask = (historical_data.index >= start_date_hist) & (historical_data.index <= end_date_hist)
        recent_data = historical_data[mask]
        
        if not recent_data.empty:
            # 绘制历史数据（实线）
            ax.plot(recent_data.index, recent_data['close'], 
                   label='Historical Price', 
                   color='gray', 
                   linestyle='-', 
                   linewidth=2,
                   marker='o')
            
            # 添加历史数据标签
            for x, y in zip(recent_data.index, recent_data['close']):
                # 确保y是数值类型并格式化
                try:
                    y_value = float(y)
                    label = "{:.2f}".format(y_value)
                except (ValueError, TypeError):
                    continue
                    
                ax.annotate(label,
                           (x, y),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           color='gray',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.5',
                                   fc='white',
                                   ec='gray',
                                   alpha=0.7))
    
    # 绘制预测数据（虚线）
    for model_name, pred_values in predictions.items():
        if len(pred_values) != len(pred_dates):
            print(f"警告: {model_name} 的预测值数量与预测日期数量不匹配")
            continue
        
        # 绘制预测线和点（虚线）
        line = ax.plot(pred_dates, pred_values, 
                      label=f'{model_name} Prediction',
                      linestyle='--',
                      marker='o')
        color = line[0].get_color()
        
        # 添加预测数据标签
        for x, y in zip(pred_dates, pred_values):
            # 确保y是数值类型并格式化
            try:
                y_value = float(y)
                label = "{:.2f}".format(y_value)
            except (ValueError, TypeError):
                continue
                
            ax.annotate(label,
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
    
    ax.set_title('Stock Price History and Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    # 自动调整x轴日期显示
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 将图表嵌入到Tkinter界面中
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
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