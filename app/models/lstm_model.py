"""
LSTM模型实现文件
该模型结合了LSTM层和注意力机制，用于时间序列预测
主要特点：
1. 使用双向LSTM进行特征提取
2. 集成了自注意力机制
3. 使用了多个正则化技术（Dropout、LayerNormalization、BatchNormalization）
4. 添加了残差连接
5. 使用了学习率调度策略
6. 支持动态参数调整
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization, Layer, BatchNormalization,
    Input, Concatenate, Add, Bidirectional, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from app.config.config import LSTM_PARAMS

class AttentionLayer(Layer):
    """
    自定义注意力层
    用于计算序列中不同时间步的重要性权重
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化注意力权重矩阵和偏置项
        self.W = self.add_weight(
            shape=(input_shape[2], input_shape[2]),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(input_shape[1],),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算注意力权重并应用到输入序列
        q = tf.matmul(inputs, self.W)  # 线性变换
        a = tf.matmul(q, inputs, transpose_b=True)  # 计算注意力分数
        a = tf.nn.softmax(a, axis=-1)  # 归一化注意力权重
        return tf.matmul(a, inputs)  # 应用注意力权重

def create_lstm_model(input_shape, params=None):
    """
    创建LSTM模型
    Args:
        input_shape: 输入数据的形状，格式为(时间步长, 特征维度)
        params: 可选的模型参数字典，用于动态调整模型结构
    Returns:
        构建好的LSTM模型
    """
    # 使用传入的参数或默认参数
    if params is None:
        params = LSTM_PARAMS
    else:
        # 合并默认参数和传入参数
        default_params = LSTM_PARAMS.copy()
        default_params.update(params)
        params = default_params
    
    # 输入层
    inputs = Input(shape=input_shape)
    
    # 第一个双向LSTM层
    x = Bidirectional(
        LSTM(params['hidden_dim'], 
             return_sequences=True,
             dropout=params.get('dropout', LSTM_PARAMS['dropout']),
             recurrent_dropout=params.get('dropout', LSTM_PARAMS['dropout'])/2)
    )(inputs)
    x = LayerNormalization()(x)
    
    # 注意力层
    attention_out = AttentionLayer()(x)
    x = Add()([x, attention_out])  # 残差连接
    x = LayerNormalization()(x)
    
    # 第二个双向LSTM层
    x = Bidirectional(
        LSTM(params['hidden_dim']//2,
             return_sequences=True,
             dropout=params.get('dropout', LSTM_PARAMS['dropout']),
             recurrent_dropout=params.get('dropout', LSTM_PARAMS['dropout'])/2)
    )(x)
    x = LayerNormalization()(x)
    
    # 第三个双向LSTM层
    x = Bidirectional(
        LSTM(params['hidden_dim']//4,
             return_sequences=True,
             dropout=params.get('dropout', LSTM_PARAMS['dropout']),
             recurrent_dropout=params.get('dropout', LSTM_PARAMS['dropout'])/2)
    )(x)
    x = LayerNormalization()(x)
    
    # 全局平均池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
    x = Dense(params['hidden_dim'], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout', LSTM_PARAMS['dropout']))(x)
    
    x = Dense(params['hidden_dim']//2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout', LSTM_PARAMS['dropout']))(x)
    
    # 输出层
    outputs = Dense(1, activation='linear')(x)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 配置优化器
    optimizer = Adam(
        learning_rate=params.get('learning_rate', LSTM_PARAMS['learning_rate']),
        clipnorm=1.0  # 添加梯度裁剪
    )
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='huber',  # 使用Huber损失函数，对异常值更鲁棒
        metrics=['mae', 'mse']
    )
    
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, progress_callback=None, **kwargs):
    """
    训练LSTM模型
    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        X_test: 测试数据特征
        y_test: 测试数据标签
        progress_callback: 训练进度回调函数
        **kwargs: 额外的模型参数，用于动态调整模型
    Returns:
        预测结果、真实值和训练好的模型
    """
    # 创建模型时传入动态参数
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
    
    # 进度回调类
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, epochs, logs.get('loss'), logs.get('val_loss'))
    
    # 配置回调函数
    callbacks = [
        # 早停策略
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率自适应调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        # 训练进度回调
        ProgressCallback() if progress_callback else None
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[cb for cb in callbacks if cb],
        verbose=1
    )
    
    # 预测并返回结果
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
