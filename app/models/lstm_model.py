"""
LSTM模型实现文件
该模型结合了LSTM层和注意力机制，用于时间序列预测
主要特点：
1. 使用多层LSTM进行特征提取
2. 集成了自注意力机制
3. 使用了多个正则化技术（Dropout、LayerNormalization、BatchNormalization）
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Layer, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class AttentionLayer(Layer):
    """
    自定义注意力层
    用于计算序列中不同时间步的重要性权重
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化注意力权重矩阵和偏置项
        self.W = self.add_weight(shape=(input_shape[2], input_shape[2]), 
                                initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],), 
                                initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算注意力权重并应用到输入序列
        q = tf.matmul(inputs, self.W)  # 线性变换
        a = tf.matmul(q, inputs, transpose_b=True)  # 计算注意力分数
        attention_weights = tf.nn.softmax(a, axis=-1)  # 归一化注意力权重
        return tf.matmul(attention_weights, inputs)  # 应用注意力权重

def create_lstm_model(input_shape):
    """
    创建LSTM模型
    Args:
        input_shape: 输入数据的形状，格式为(时间步长, 特征维度)
    Returns:
        构建好的LSTM模型
    """
    model = Sequential([
        # 第一层LSTM，返回序列用于后续处理
        LSTM(units=256, return_sequences=True, input_shape=input_shape),
        LayerNormalization(),  # 层归一化，用于稳定训练
        Dropout(0.1),  # dropout防止过拟合
        
        # 注意力层，用于关注重要的时间步
        AttentionLayer(),
        LayerNormalization(),
        
        # 第二层LSTM，进一步提取高级特征
        LSTM(units=128, return_sequences=True),
        LayerNormalization(),
        Dropout(0.1),
        
        # 第三层LSTM，输出单个时间步的特征
        LSTM(units=64, return_sequences=False),
        LayerNormalization(),
        Dropout(0.1),
        
        # 全连接层，逐步降维到最终预测值
        Dense(units=128, activation='relu'),
        BatchNormalization(),
        Dense(units=64, activation='relu'),
        BatchNormalization(),
        Dense(units=32, activation='relu'),
        BatchNormalization(),
        Dense(units=1, activation='linear')  # 输出层，线性激活用于回归任务
    ])
    
    # 配置优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, progress_callback=None):
    """
    训练LSTM模型
    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        X_test: 测试数据特征
        y_test: 测试数据标签
        progress_callback: 训练进度回调函数
    Returns:
        预测结果、真实值和训练好的模型
    """
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 进度回调类，用于显示训练进度
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, 100)
    
    # 配置回调函数
    callbacks = [
        # 早停策略，防止过拟合
        EarlyStopping(
            monitor='val_loss',
            patience=20,  # 20轮无改善则停止
            restore_best_weights=True,  # 恢复最佳权重
            verbose=1
        ),
        # 学习率自适应调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # 学习率衰减因子
            patience=10,  # 10轮无改善则降低学习率
            min_lr=1e-6,  # 最小学习率
            verbose=1
        ),
        ProgressCallback() if progress_callback else None
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[cb for cb in callbacks if cb],
        verbose=1
    )
    
    # 预测并返回结果
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
