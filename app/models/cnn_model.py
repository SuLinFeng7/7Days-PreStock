"""
CNN模型实现文件
使用一维卷积神经网络进行时间序列预测
主要特点：
1. 使用多层一维卷积提取时序特征
2. 采用残差连接和批归一化提高模型稳定性
3. 使用Huber损失函数提高对异常值的鲁棒性
4. 支持动态参数调整
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def create_cnn_model(input_shape, params=None):
    """
    创建一维CNN模型
    Args:
        input_shape: 输入数据的形状，格式为(时间步长, 特征维度)
        params: 可选的模型参数字典，用于动态调整模型结构
    Returns:
        构建好的CNN模型
    """
    # 使用传入的参数或默认参数
    if params is None:
        params = {
            'filters': 64,
            'learning_rate': 0.001,
            'dropout': 0.2
        }
    
    model = Sequential([
        # 第一个卷积块
        Conv1D(128, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # 第二个卷积块
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # 第三个卷积块
        Conv1D(32, 3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)  # 输出层，用于回归预测
    ])
    
    # 配置优化器和损失函数
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
    model.compile(
        optimizer=optimizer, 
        loss='huber',  # 使用Huber损失函数，对异常值更鲁棒
        metrics=['mae', 'mse']
    )
    return model

def train_cnn_model(X_train, y_train, X_test, y_test, progress_callback=None, **kwargs):
    """
    训练CNN模型
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
    model = create_cnn_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        params=kwargs
    )
    
    # 获取训练参数
    epochs = kwargs.get('epochs', 200)
    batch_size = kwargs.get('batch_size', 64)
    patience = kwargs.get('patience', 15)
    
    # 进度回调类
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, epochs, logs.get('loss'), logs.get('val_loss'))
    
    # 配置回调函数
    callbacks = [
        # 早停策略，防止过拟合
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,  # 恢复最佳权重
            verbose=1
        ),
        # 学习率自适应调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
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
    
    # 生成预测结果
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
