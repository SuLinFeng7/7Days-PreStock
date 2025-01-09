# Tushare API配置
TUSHARE_TOKEN = '0648c566c4cd4d547549ef08dfe17ba164b1e30c50f3732730f9093b'

# 模型参数配置
TIMESTEPS = 60  # 时间步长
TRAIN_YEARS = 3  # 训练数据年限

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
    'epochs': 50,
    'batch_size': 32
}
