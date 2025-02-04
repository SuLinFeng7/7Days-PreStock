import numpy as np
from .lstm_model import train_lstm_model
from .cnn_model import train_cnn_model
from app.data.data_preprocessing import prepare_prediction_data
from app.utils.utils import (
    calculate_metrics,
    calculate_mape,
    calculate_rmse,
    calculate_mae
)
from .transformer_model import train_transformer_model
import torch
from app.config.config import PREDICTION_DAYS
from datetime import datetime

class ModelTrainer:
    def __init__(self, data, progress_bar=None):
        self.data = data
        self.progress_bar = progress_bar
        self.models = {
            'LSTM': {'func': train_lstm_model, 'epochs': 70, 'max_retries': 3},
            'CNN': {'func': train_cnn_model, 'epochs': 100, 'max_retries': 3},
            'Transformer': {'func': train_transformer_model, 'epochs': 150, 'max_retries': 3}
        }
        self.total_steps = sum(model['epochs'] for model in self.models.values())
        self.current_step = 0
        self.current_model_epochs = 0
        self.default_prediction_days = PREDICTION_DAYS
        self.training_times = {}  # 添加训练时间记录字典
        
        # 添加预测误差阈值
        self.mape_threshold = 20  # MAPE阈值为20%
        self.rmse_threshold = 2.0  # RMSE阈值
        self.consecutive_bad_predictions = 0  # 连续不良预测计数
        self.max_consecutive_bad = 5  # 最大连续不良预测次数

    def evaluate_prediction_quality(self, y_true, y_pred):
        """评估预测质量，返回是否需要调整模型"""
        mape = calculate_mape(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        
        if mape > self.mape_threshold or rmse > self.rmse_threshold:
            self.consecutive_bad_predictions += 1
        else:
            self.consecutive_bad_predictions = 0
            
        return self.consecutive_bad_predictions >= self.max_consecutive_bad

    def adjust_model_parameters(self, model_name, retry_count):
        """根据重试次数调整模型参数"""
        if model_name == 'LSTM':
            # LSTM参数调整策略
            self.models[model_name]['epochs'] = min(100, 70 + retry_count * 10)
            return {
                'learning_rate': 0.001 / (1 + retry_count * 0.5),
                'batch_size': 32 * (retry_count + 1),
                'dropout': 0.2 + retry_count * 0.1
            }
        elif model_name == 'Transformer':
            # Transformer参数调整策略
            self.models[model_name]['epochs'] = min(200, 150 + retry_count * 15)
            return {
                'learning_rate': 0.0001 / (1 + retry_count * 0.5),
                'batch_size': 16 * (retry_count + 1),
                'n_heads': 4 + retry_count,
                'dropout': 0.1 + retry_count * 0.05
            }
        else:  # CNN
            # CNN参数调整策略
            self.models[model_name]['epochs'] = min(150, 100 + retry_count * 10)
            return {
                'learning_rate': 0.001 / (1 + retry_count * 0.5),
                'batch_size': 32 * (retry_count + 1),
                'filters': 32 * (retry_count + 1)
            }

    def update_progress(self, completed_epochs, total_epochs, train_loss=None, val_loss=None):
        """更新训练进度
        Args:
            completed_epochs: 已完成的训练轮数
            total_epochs: 总训练轮数
            train_loss: 训练损失
            val_loss: 验证损失
        """
        if self.progress_bar:
            try:
                current_model_progress = completed_epochs / total_epochs
                total_progress = ((self.current_step + current_model_progress * self.current_model_epochs) 
                                * 100 / self.total_steps)
                self.progress_bar['value'] = float(total_progress)
                self.progress_bar.update()
                
                # 添加损失值的显示
                if train_loss is not None and val_loss is not None:
                    print(f"\r训练进度: {total_progress:.1f}% - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}", end='\r')
                else:
                    print(f"\r训练进度: {total_progress:.1f}%", end='\r')
                
            except Exception as e:
                print(f"更新进度条时出错: {str(e)}")

    def train_all_models(self):
        try:
            predictions = {}
            metrics = {}
            trained_models = {}
            self.current_step = 0
            
            # 获取训练数据
            X_train, y_train, X_test, y_test, price_scaler, feature_scaler = prepare_prediction_data(self.data)
            
            # 确保数据类型正确
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.float32)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)

            for model_name, model_info in self.models.items():
                retry_count = 0
                model_trained = False
                
                while not model_trained and retry_count < model_info['max_retries']:
                    try:
                        print(f"\n开始训练 {model_name} 模型... (第 {retry_count + 1} 次尝试)")
                        self.current_model_epochs = int(model_info['epochs'])
                        
                        # 获取调整后的参数
                        adjusted_params = self.adjust_model_parameters(model_name, retry_count)
                        print(f"使用参数: {adjusted_params}")
                        
                        # 记录开始时间
                        start_time = datetime.now()
                        
                        # 训练模型
                        y_pred, y_true, model = model_info['func'](
                            X_train, y_train, X_test, y_test, 
                            self.update_progress,
                            **adjusted_params
                        )
                        
                        # 检查模型训练是否成功
                        if y_pred is None or y_true is None or model is None:
                            print(f"\n{model_name} 模型训练失败，尝试重新训练...")
                            retry_count += 1
                            if retry_count >= model_info['max_retries']:
                                print(f"{model_name} 模型达到最大重试次数，跳过该模型")
                                continue
                            continue
                        
                        # 记录结束时间并计算训练时长
                        end_time = datetime.now()
                        training_duration = (end_time - start_time).total_seconds() / 60
                        self.training_times[model_name] = training_duration
                        print(f"\n{model_name} 模型训练完成，耗时: {training_duration:.2f} 分钟")
                        
                        # 保存训练好的模型
                        trained_models[model_name] = model
                        model_trained = True
                        
                        # 生成未来预测
                        last_sequence = X_test[-1:].copy()
                        future_predictions = []
                        
                        print(f"生成 {model_name} 模型的未来预测...")
                        # 预测未来30天
                        for day in range(self.default_prediction_days):
                            if model_name == 'Transformer':
                                with torch.no_grad():
                                    last_sequence_tensor = torch.FloatTensor(last_sequence)
                                    next_pred = model(last_sequence_tensor)[0]
                                    next_pred = next_pred.numpy()
                            else:
                                next_pred = model.predict(last_sequence, verbose=0)
                            
                            if isinstance(next_pred, np.ndarray):
                                next_pred = next_pred.item()
                            
                            pred_reshaped = np.array([[next_pred]], dtype=np.float32)
                            actual_price = price_scaler.inverse_transform(pred_reshaped)[0, 0]
                            future_predictions.append(float(actual_price))
                            
                            last_sequence = np.roll(last_sequence, -1, axis=1)
                            last_sequence[0, -1, 0] = float(next_pred)
                            
                            if day % 5 == 0:
                                print(f"第{day+1}天预测值: {actual_price:.2f}")
                        
                        predictions[f"{model_name}_future"] = future_predictions
                        
                        # 计算评估指标
                        mape = calculate_mape(y_true, y_pred)
                        rmse = calculate_rmse(y_true, y_pred)
                        mae = calculate_mae(y_true, y_pred)
                        
                        metrics[f"{model_name}_future"] = {
                            'MAPE': float(mape),
                            'RMSE': float(rmse),
                            'MAE': float(mae),
                            'retry_count': retry_count
                        }
                        
                    except Exception as e:
                        print(f"{model_name} 模型训练失败: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        retry_count += 1
                        if retry_count >= model_info['max_retries']:
                            print(f"{model_name} 模型训练失败，已达到最大重试次数")
                            continue
                
                # 更新总进度
                self.current_step += self.current_model_epochs

            # 计算最近两年的预测准确度
            historical_predictions = {}
            for model_name, model in trained_models.items():
                if model_name == 'Transformer':
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(X_test)
                        y_pred, _ = model(X_test_tensor)
                        y_pred = y_pred.numpy()
                else:
                    y_pred = model.predict(X_test, verbose=0)
                
                y_pred_actual = price_scaler.inverse_transform(y_pred.reshape(-1, 1))
                y_test_actual = price_scaler.inverse_transform(y_test.reshape(-1, 1))
                
                historical_predictions[f"{model_name}_historical"] = {
                    'predicted': y_pred_actual.flatten(),
                    'actual': y_test_actual.flatten()
                }
                
                # 计算评估指标
                metrics[f"{model_name}_historical"] = {
                    'MAPE': float(calculate_mape(y_test_actual, y_pred_actual)),
                    'RMSE': float(calculate_rmse(y_test_actual, y_pred_actual)),
                    'MAE': float(calculate_mae(y_test_actual, y_pred_actual)),
                    'training_time': self.training_times[model_name]
                }
            
            # 打印所有模型的训练时长和重试次数
            print("\n各模型训练结果:")
            for model_name, duration in self.training_times.items():
                retry_count = metrics.get(f"{model_name}_future", {}).get('retry_count', 0)
                print(f"{model_name}: 训练时长 {duration:.2f} 分钟, 重试次数 {retry_count}")
            
            return predictions, historical_predictions, metrics
            
        except Exception as e:
            print(f"模型训练过程中发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e