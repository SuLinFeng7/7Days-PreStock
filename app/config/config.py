# Tushare API配置
TUSHARE_TOKEN = '0648c566c4cd4d547549ef08dfe17ba164b1e30c50f3732730f9093b'

# 模型参数配置
TIMESTEPS = 60  # 时间步长
TRAIN_YEARS = 10  # 总数据年限改为10年
TRAIN_TEST_SPLIT = 0.8  # 训练集占比80%
PREDICTION_DAYS = 30  # 未来预测天数

# 模型训练参数
LSTM_PARAMS = {
    'hidden_dim': 128,        # 增加隐藏层维度
    'num_layers': 3,          # 保持3层LSTM
    'dropout': 0.2,           # 添加dropout参数
    'bidirectional': True,    # 使用双向LSTM
    'epochs': 100,            # 增加训练轮数
    'batch_size': 32,         # 保持批次大小
    'learning_rate': 0.001,   # 保持学习率
    'optimizer': 'adam',      # 使用Adam优化器
    'patience': 15,           # 早停耐心值
    'reduce_lr_patience': 8,  # 学习率衰减耐心值
    'reduce_lr_factor': 0.5,  # 学习率衰减因子
    'min_lr': 1e-6           # 最小学习率
}

CNN_PARAMS = {
    'filters': 128,           # 第一层卷积 filters 增加到 128
    'kernel_size': 3,         # 卷积核大小保持不变
    'dropout': 0.3,           # Dropout 率增加到 0.3
    'epochs': 200,            # 训练轮数
    'batch_size': 32,         # 批量大小
    'learning_rate': 0.001,   # 学习率
    'patience': 15,           # 早停耐心值
    'reduce_lr_patience': 8,  # 学习率衰减耐心值
    'reduce_lr_factor': 0.5,  # 学习率衰减因子
    'min_lr': 1e-6,           # 最小学习率
    'optimizer': 'adamw',     # 使用 AdamW 优化器
    'loss': 'huber'           # 使用 Huber 损失函数
}

TRANSFORMER_PARAMS = {
    'd_model': 128,          # 模型维度
    'd_ff': 512,            # 前馈网络维度
    'num_layers': 3,        # 编码器层数
    'n_heads': 8,           # 注意力头数
    'dropout': 0.15,        # dropout率
    'attention_dropout': 0.1, # 注意力dropout率
    'epochs': 150,          # 训练轮数
    'batch_size': 64,       # 批次大小
    'warmup_steps': 4000,   # 预热步数
    'learning_rate': 0.0001, # 学习率
    'weight_decay': 0.01,   # 权重衰减
    'clip_grad_norm': 1.0,  # 梯度裁剪阈值
    'loss': 'huber'         # 损失函数类型
}
