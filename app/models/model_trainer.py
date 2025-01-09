import numpy as np
from .lstm_model import train_lstm_model
from .cnn_model import train_cnn_model
from data.data_preprocessing import prepare_prediction_data
from utils.utils import (
    calculate_metrics,
    calculate_mape,
    calculate_rmse,
    calculate_mae
)
from .transformer_model import train_transformer_model
import torch

class ModelTrainer:
    def __init__(self, data, progress_bar=None):
        self.data = data
        self.progress_bar = progress_bar
        self.models = {
            'LSTM': {'func': train_lstm_model, 'epochs': 70},
            'CNN': {'func': train_cnn_model, 'epochs': 100},
            'Transformer': {'func': train_transformer_model, 'epochs': 30}
        }
        self.total_steps = sum(model['epochs'] for model in self.models.values())
        self.current_step = 0
        self.current_model_epochs = 0

    def update_progress(self, completed_epochs, total_epochs):
        if self.progress_bar:
            try:
                current_model_progress = completed_epochs / total_epochs
                total_progress = ((self.current_step + current_model_progress * self.current_model_epochs) 
                                * 100 / self.total_steps)
                self.progress_bar['value'] = float(total_progress)
                self.progress_bar.update()
                print(f"训练进度: {total_progress:.1f}%", end='\r')
            except Exception as e:
                print(f"更新进度条时出错: {str(e)}")

    def train_all_models(self):
        try:
            predictions = {}
            metrics = {}
            self.current_step = 0
            failed_models = []

            X_train, y_train, X_test, y_test, scaler = prepare_prediction_data(self.data)
            
            # 确保数据类型正确
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.float32)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)

            for model_name, model_info in self.models.items():
                try:
                    print(f"\n开始训练 {model_name} 模型...")
                    self.current_model_epochs = int(model_info['epochs'])
                    
                    # 训练模型
                    y_pred, y_true, model = model_info['func'](
                        X_train, y_train, X_test, y_test, 
                        self.update_progress
                    )
                    
                    # 生成未来预测
                    last_sequence = X_test[-1:].copy()
                    future_predictions = []
                    
                    print(f"生成 {model_name} 模型的未来预测...")
                    # 预测未来7天
                    for day in range(7):
                        # 根据模型类型选择预测方法
                        if model_name == 'Transformer':
                            # Transformer模型使用PyTorch
                            with torch.no_grad():
                                last_sequence_tensor = torch.FloatTensor(last_sequence)
                                next_pred, _ = model(last_sequence_tensor)
                                next_pred = next_pred.numpy()
                        else:
                            # LSTM和CNN模型使用TensorFlow
                            next_pred = model.predict(last_sequence, verbose=0)
                        
                        # 确保预测值是标量
                        if isinstance(next_pred, np.ndarray):
                            next_pred = next_pred.item()
                        
                        # 将预测值转换回实际价格
                        pred_reshaped = np.array([[next_pred]], dtype=np.float32)
                        actual_price = scaler.inverse_transform(pred_reshaped)[0, 0]
                        future_predictions.append(float(actual_price))
                        
                        # 更新序列用于下一次预测
                        last_sequence = np.roll(last_sequence, -1, axis=1)
                        last_sequence[0, -1, 0] = float(next_pred)
                        
                        print(f"第{day+1}天预测值: {actual_price:.2f}")
                    
                    predictions[model_name] = future_predictions
                    
                    # 计算评估指标
                    mape = calculate_mape(y_true, y_pred)
                    rmse = calculate_rmse(y_true, y_pred)
                    mae = calculate_mae(y_true, y_pred)
                    
                    metrics[model_name] = {
                        'MAPE': float(mape),
                        'RMSE': float(rmse),
                        'MAE': float(mae)
                    }
                    
                    print(f"{model_name} 模型训练完成")
                    
                except Exception as e:
                    print(f"{model_name} 模型训练失败: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    failed_models.append(model_name)
                    continue
                
                # 更新总进度
                self.current_step += self.current_model_epochs

            if failed_models:
                print(f"\n警告: 以下模型训练失败: {', '.join(failed_models)}")
            
            return predictions, metrics 
            
        except Exception as e:
            print(f"模型训练过程中发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e 