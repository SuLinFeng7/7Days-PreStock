# Tushare API配置
TUSHARE_TOKEN = '0648c566c4cd4d547549ef08dfe17ba164b1e30c50f3732730f9093b'

# 模型参数配置
TIMESTEPS = 60  # 时间步长
TRAIN_YEARS = 10  # 总数据年限改为10年
TRAIN_TEST_SPLIT = 0.8  # 训练集占比80%
PREDICTION_DAYS = 30  # 未来预测天数

# 模型训练参数
LSTM_PARAMS = {
    'hidden_dim': 64, # 隐藏层维度
    'num_layers': 3, # 层数
    'epochs': 50, # 训练轮数
    'batch_size': 32, # 批量大小
    'learning_rate': 0.001 # 学习率
}

CNN_PARAMS = {
    'epochs': 200, # 训练轮数
    'batch_size': 32, # 批量大小
    'early_stopping_patience': 30, # 早停步数
    'reduce_lr_patience': 10, # 学习率减少步数
    'reduce_lr_factor': 0.5, # 学习率减少因子
    'min_lr': 1e-6 # 最小学习率
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
