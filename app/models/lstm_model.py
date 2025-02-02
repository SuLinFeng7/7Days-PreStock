"""
LSTM模型实现文件
该模型结合了LSTM层和注意力机制，用于时间序列预测
主要特点：
1. 使用双向LSTM进行特征提取
2. 集成了多种注意力机制（自注意力、多尺度注意力）
3. 使用了多个正则化技术
4. 添加了残差连接
5. 使用了学习率调度策略
6. 支持动态参数调整
7. 添加了多尺度特征提取
8. 实现了突发事件处理机制
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization, Layer, BatchNormalization,
    Input, Concatenate, Add, Bidirectional, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, AveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from app.config.config import LSTM_PARAMS

class MultiScaleAttention(Layer):
    """
    多尺度注意力层
    通过不同尺度的卷积和池化操作捕捉不同时间尺度的特征
    """
    def __init__(self, **kwargs):
        super(MultiScaleAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # 不同尺度的卷积核
        self.conv1 = Conv1D(filters=32, kernel_size=1, padding='same')
        self.conv3 = Conv1D(filters=32, kernel_size=3, padding='same')
        self.conv5 = Conv1D(filters=32, kernel_size=5, padding='same')
        
        # 修改注意力权重的维度
        self.attention_dense = Dense(32)
        self.attention_weights = self.add_weight(
            shape=(3, 32, 32),
            initializer='glorot_uniform',
            trainable=True,
            name='scale_attention_weights'
        )
        
        super(MultiScaleAttention, self).build(input_shape)
    
    def call(self, inputs):
        # 不同尺度的特征提取
        scale1 = self.conv1(inputs)  # shape: (batch_size, time_steps, 32)
        scale3 = self.conv3(inputs)  # shape: (batch_size, time_steps, 32)
        scale5 = self.conv5(inputs)  # shape: (batch_size, time_steps, 32)
        
        # 转换输入维度
        query = self.attention_dense(inputs)  # shape: (batch_size, time_steps, 32)
        
        # 计算注意力权重
        scales = [scale1, scale3, scale5]
        weighted_scales = []
        
        for i, scale in enumerate(scales):
            # 计算注意力分数
            attention_score = tf.einsum('bti,ij->btj', query, self.attention_weights[i])
            attention_weights = tf.nn.softmax(attention_score, axis=-1)
            # 应用注意力权重
            weighted_scale = scale * attention_weights
            weighted_scales.append(weighted_scale)
        
        # 合并不同尺度的特征
        output = tf.add_n(weighted_scales) / 3.0
        
        return output
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 32)

class EventAttention(Layer):
    """
    突发事件注意力层
    用于检测和处理时间序列中的异常和突发变化
    """
    def __init__(self, threshold=3.0, **kwargs):
        super(EventAttention, self).__init__(**kwargs)
        self.threshold = threshold
    
    def build(self, input_shape):
        self.event_dense = Dense(input_shape[-1])
        self.attention_weights = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='event_attention_weights'
        )
        super(EventAttention, self).build(input_shape)
    
    def call(self, inputs):
        # 计算移动平均和标准差
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        
        # 检测异常值
        z_scores = tf.abs(inputs - mean) / (std + 1e-6)
        event_mask = tf.cast(z_scores > self.threshold, tf.float32)
        
        # 对异常值特别关注
        event_features = self.event_dense(inputs)
        attention_scores = tf.matmul(event_features, self.attention_weights)
        attention_weights = tf.nn.softmax(attention_scores * event_mask)
        
        return inputs * attention_weights

class ContinuityRegularization(Layer):
    """
    连续性正则化层
    用于确保预测结果的连续性，避免剧烈波动
    """
    def __init__(self, smoothing_factor=0.5, **kwargs):
        super(ContinuityRegularization, self).__init__(**kwargs)
        self.smoothing_factor = smoothing_factor
    
    def call(self, inputs):
        # 使用 tf.keras.backend 操作替代直接的 tensorflow 操作
        # 计算相邻时间步的差值
        diffs = inputs[:, 1:] - inputs[:, :-1]
        # 应用平滑因子
        smoothed = inputs[:, :-1] + diffs * self.smoothing_factor
        # 保持最后一个预测值不变
        last_value = inputs[:, -1:]
        # 使用 tf.keras.layers.Concatenate 替代 tf.concat
        concat_layer = tf.keras.layers.Concatenate(axis=1)
        return concat_layer([smoothed, last_value])

def create_lstm_model(input_shape, params=None):
    """
    创建增强版LSTM模型
    """
    if params is None:
        params = LSTM_PARAMS
    else:
        default_params = LSTM_PARAMS.copy()
        default_params.update(params)
        params = default_params
    
    # 输入层
    inputs = Input(shape=input_shape)
    
    # 多尺度特征提取
    x = MultiScaleAttention()(inputs)
    
    # 第一个双向LSTM层
    lstm1 = Bidirectional(
        LSTM(params['hidden_dim'], 
             return_sequences=True,
             dropout=params.get('dropout', LSTM_PARAMS['dropout']),
             recurrent_dropout=params.get('dropout', LSTM_PARAMS['dropout'])/2,
             kernel_regularizer=tf.keras.regularizers.l2(0.01))
    )(x)
    lstm1 = LayerNormalization()(lstm1)
    
    # 突发事件注意力
    event_attention = EventAttention()(lstm1)
    x = Add()([lstm1, event_attention])
    x = LayerNormalization()(x)
    
    # 第二个双向LSTM层
    lstm2 = Bidirectional(
        LSTM(params['hidden_dim']//2,
             return_sequences=True,
             dropout=params.get('dropout', LSTM_PARAMS['dropout']),
             recurrent_dropout=params.get('dropout', LSTM_PARAMS['dropout'])/2,
             kernel_regularizer=tf.keras.regularizers.l2(0.01))
    )(x)
    lstm2 = LayerNormalization()(lstm2)
    
    # 多尺度特征融合
    x = Concatenate()([
        GlobalAveragePooling1D()(lstm2),
        MaxPooling1D(pool_size=2)(lstm2)[:, -1, :],
        AveragePooling1D(pool_size=2)(lstm2)[:, -1, :]
    ])
    
    # 全连接层
    x = Dense(params['hidden_dim'], 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout', LSTM_PARAMS['dropout']))(x)
    
    x = Dense(params['hidden_dim']//2, 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout', LSTM_PARAMS['dropout']))(x)
    
    # 输出层
    outputs = Dense(1, activation='linear')(x)
    # 使用 Lambda 层包装 expand_dims 操作
    outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(outputs)
    outputs = ContinuityRegularization()(outputs)
    outputs = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(outputs)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 配置优化器
    optimizer = Adam(
        learning_rate=params.get('learning_rate', LSTM_PARAMS['learning_rate']) * 0.1,
        clipnorm=0.5
    )
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, progress_callback=None, **kwargs):
    """
    训练LSTM模型
    """
    try:
        # 检查输入数据
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise ValueError("训练数据包含 NaN 值")
        if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
            raise ValueError("训练数据包含无限值")
        
        # 添加数据增强
        noise_scale = 0.001  # 很小的噪声
        X_train_noisy = X_train + np.random.normal(0, noise_scale, X_train.shape)
        y_train_noisy = y_train + np.random.normal(0, noise_scale, y_train.shape)
        
        # 合并原始数据和噪声数据
        X_train_combined = np.concatenate([X_train, X_train_noisy], axis=0)
        y_train_combined = np.concatenate([y_train, y_train_noisy], axis=0)
            
        # 创建模型
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            params=kwargs
        )
        
        # 获取训练参数
        epochs = kwargs.get('epochs', LSTM_PARAMS['epochs'])
        batch_size = kwargs.get('batch_size', LSTM_PARAMS['batch_size'])
        patience = kwargs.get('patience', LSTM_PARAMS['patience'])
        reduce_lr_factor = kwargs.get('reduce_lr_factor', LSTM_PARAMS['reduce_lr_factor'])
        reduce_lr_patience = kwargs.get('reduce_lr_patience', LSTM_PARAMS['reduce_lr_patience'])
        min_lr = kwargs.get('min_lr', LSTM_PARAMS['min_lr'])
        
        # 配置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=1
            )
        ]
        
        if progress_callback:
            callbacks.append(tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_callback(epoch + 1, epochs, logs.get('loss'), logs.get('val_loss'))
            ))
        
        # 训练模型
        history = model.fit(
            X_train_combined, y_train_combined,  # 使用增强后的数据
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 确保预测结果和真实值长度一致
        min_len = min(len(y_test), len(y_pred))
        y_pred = y_pred[:min_len]
        y_test = y_test[:min_len]
        
        # 后处理：使用移动平均平滑预测结果
        window_size = 3
        y_pred_smoothed = np.convolve(y_pred.flatten(), 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        # 补齐长度
        padding = np.repeat(y_pred_smoothed[0], (window_size-1)//2)
        y_pred_smoothed = np.concatenate([padding, y_pred_smoothed])
        
        # 确保最终长度一致
        y_pred_smoothed = y_pred_smoothed[:len(y_test)]
        
        # 检查预测结果
        if np.any(np.isnan(y_pred_smoothed)):
            raise ValueError("预测结果包含 NaN 值")
        if np.any(np.isinf(y_pred_smoothed)):
            raise ValueError("预测结果包含无限值")
            
        # 检查预测值的范围
        if np.any(np.abs(y_pred_smoothed) > 1e6):
            print("警告：预测值范围异常，可能需要检查数据标准化过程")
            
        return y_pred_smoothed.reshape(-1, 1), y_test.reshape(-1, 1), model
        
    except Exception as e:
        print(f"LSTM模型训练失败: {str(e)}")
        print("详细错误信息:")
        import traceback
        print(traceback.format_exc())
        return None, None, None
