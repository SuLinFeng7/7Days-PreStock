from utils.record_keeper import RecordKeeper
from utils.visualization import create_prediction_chart
import pandas as pd
from datetime import datetime, timedelta
import tkinter as tk
import matplotlib.pyplot as plt

def test_record():
    """测试记录保存功能"""
    keeper = RecordKeeper()
    success = keeper.test_write()
    
    if success:
        print("\n记录测试成功！文件写入正常。")
    else:
        print("\n记录测试失败！请检查错误信息。")
    return success

def test_chart():
    """测试图表生成功能"""
    try:
        print("\n开始测试图表生成...")
        
        # 创建测试窗口
        root = tk.Tk()
        root.geometry("1000x800")  # 设置窗口大小
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        print("创建测试数据...")
        # 1. 预测数据
        predictions = {
            'LSTM': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            'CNN': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        }
        
        # 2. 历史数据
        print("生成历史数据...")
        dates = pd.date_range(end=datetime.now(), periods=7)
        historical_data = pd.DataFrame({
            'close': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0]
        }, index=dates)
        print(f"历史数据:\n{historical_data}")
        
        # 3. 预测日期范围
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=6)
        print(f"预测日期范围: {start_date} 到 {end_date}")
        
        print("开始生成图表...")
        try:
            # 尝试生成图表
            chart_widget = create_prediction_chart(
                predictions=predictions,
                start_date=start_date,
                end_date=end_date,
                parent_frame=frame,
                historical_data=historical_data
            )
            
            print("图表生成完成，检查结果...")
            if chart_widget:
                print("图表组件创建成功！")
                chart_widget.pack(fill=tk.BOTH, expand=True)
                
                # 显示图表5秒后关闭
                print("显示图表5秒...")
                root.after(5000, root.destroy)
                root.mainloop()
                return True
            else:
                print("图表组件创建失败！")
                return False
                
        except Exception as e:
            print(f"生成图表时发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            plt.close('all')  # 清理所有图表
            return False
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        plt.close('all')  # 清理所有图表
        return False
    finally:
        try:
            plt.close('all')  # 确保清理所有图表
        except:
            pass

def main():
    print("开始测试...")
    
    # 测试记录功能
    print("\n1. 测试记录功能")
    record_success = test_record()
    print(f"记录测试结果: {'成功' if record_success else '失败'}")
    
    # 测试图表功能
    print("\n2. 测试图表功能")
    chart_success = test_chart()
    print(f"图表测试结果: {'成功' if chart_success else '失败'}")
    
    # 总体测试结果
    print("\n测试总结:")
    if record_success and chart_success:
        print("所有测试通过！✅")
    else:
        print("测试未完全通过，请检查错误信息。❌")
        if not record_success:
            print("- 记录功能测试失败")
        if not chart_success:
            print("- 图表功能测试失败")

if __name__ == "__main__":
    main() 