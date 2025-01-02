import numpy as np
from .lstm_model import train_lstm_model
from .cnn_model import train_cnn_model
from data.data_preprocessing import prepare_prediction_data
from utils.utils import calculate_metrics

class ModelTrainer:
    def __init__(self, data, progress_bar=None):
        self.data = data
        self.progress_bar = progress_bar
        self.models = {
            'LSTM': {'func': train_lstm_model, 'epochs': 70},
            'CNN': {'func': train_cnn_model, 'epochs': 200}
        }
        self.total_steps = sum(model['epochs'] for model in self.models.values())
        self.current_step = 0

    def update_progress(self, completed_epochs, total_epochs):
        if self.progress_bar:
            current_model_progress = completed_epochs / total_epochs
            total_progress = ((self.current_step + current_model_progress * self.current_model_epochs) 
                             * 100 / self.total_steps)
            self.progress_bar['value'] = total_progress
            self.progress_bar.update()

    def train_all_models(self):
        predictions = {}
        metrics = {}
        self.current_step = 0

        X_train, y_train, X_test, y_test, scaler = prepare_prediction_data(self.data)

        for model_name, model_info in self.models.items():
            self.current_model_epochs = model_info['epochs']
            # 训练模型
            y_pred, y_true, model = model_info['func'](
                X_train, y_train, X_test, y_test, 
                self.update_progress
            )
            
            # 计算评估指标
            mape, rmse, mae = calculate_metrics(y_true, y_pred)
            metrics[model_name] = {
                'MAPE': mape,
                'RMSE': rmse,
                'MAE': mae
            }
            
            # 生成未来预测
            last_sequence = X_test[-1:]
            future_predictions = []
            
            # 预测未来7天
            current_sequence = last_sequence.copy()
            for _ in range(7):
                # 预测下一个值
                next_pred = model.predict(current_sequence)
                # 将预测值转换回实际价格
                pred_reshaped = next_pred.reshape(-1, 1)  # 确保是2D数组
                actual_price = scaler.inverse_transform(pred_reshaped)[0, 0]
                future_predictions.append(actual_price)
                
                # 更新序列用于下一次预测
                current_sequence = np.roll(current_sequence, -1, axis=1)
                # 只更新最后一个时间步的第一个特征（价格）
                current_sequence[0, -1, 0] = next_pred[0]
            
            predictions[model_name] = future_predictions
            
            # 更新总进度
            self.current_step += self.current_model_epochs

        return predictions, metrics 