"""
CNN模型实现文件
使用一维卷积神经网络进行时间序列预测
主要特点：
1. 使用多层一维卷积提取时序特征
2. 采用残差连接和批归一化提高模型稳定性
3. 使用Huber损失函数提高对异常值的鲁棒性
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def create_cnn_model(input_shape):
    """
    创建一维CNN模型
    Args:
        input_shape: 输入数据的形状，格式为(时间步长, 特征维度)
    Returns:
        构建好的CNN模型
    """
    model = Sequential([
        # 第一个卷积块
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),  # 批归一化层，加速训练并提高稳定性
        MaxPooling1D(pool_size=2),  # 最大池化，减少序列长度并保留重要特征
        
        # 第二个卷积块
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第三个卷积块
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),  # 全局平均池化，自适应处理不同长度的序列
        
        # 全连接层
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),  # dropout防止过拟合
        Dense(1)  # 输出层，用于回归预测
    ])
    
    # 配置优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')  # 使用Huber损失函数，对异常值更鲁棒
    return model

def train_cnn_model(X_train, y_train, X_test, y_test, progress_callback=None):
    """
    训练CNN模型
    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        X_test: 测试数据特征
        y_test: 测试数据标签
        progress_callback: 训练进度回调函数
    Returns:
        预测结果、真实值和训练好的模型
    """
    model = create_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 进度回调类
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, 200)
    
    # 配置回调函数
    callbacks = [
        # 早停策略，防止过拟合
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # 15轮无改善则停止训练
            restore_best_weights=True  # 恢复最佳权重
        ),
        ProgressCallback() if progress_callback else None
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=200,  # 最大训练轮数
        batch_size=64,  # 批次大小
        validation_data=(X_test, y_test),
        callbacks=[cb for cb in callbacks if cb],
        verbose=1
    )
    
    # 生成预测结果
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
