"""
Transformer模型实现文件
基于Transformer架构的时间序列预测模型
主要特点：
1. 使用自注意力机制捕捉时间序列中的长期依赖关系
2. 包含位置编码以保留序列顺序信息
3. 多头注意力机制实现多角度特征提取
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from config.config import TRANSFORMER_PARAMS

# 模型超参数配置
d_model = 128  # 模型隐藏层维度
d_ff = 256     # 前馈网络维度
d_k = d_v = 32  # 注意力机制中的key和value维度
n_layers = 3    # Transformer编码器层数
n_heads = 8     # 多头注意力的头数
feature_dim = 15  # 输入特征维度

class PositionalEncoding(nn.Module):
    """
    位置编码层
    为序列中的每个位置添加位置信息，使模型能够感知序列中元素的相对或绝对位置
    Args:
        d_model: 模型的隐藏层维度
        dropout: dropout率
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [seq_len, batch_size, d_model]
        Returns:
            添加位置编码后的序列
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    """
    Transformer模型主体
    使用Transformer编码器架构进行时间序列预测
    """
    def __init__(self):
        super(Transformer, self).__init__()
        # 输入特征映射层
        self.input_embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层配置
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True  # 设置batch维度在前
        )
        # 堆叠多层编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_layers
        )
        
        # 输出预测层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入数据 [batch_size, seq_len, feature_dim]
        Returns:
            预测值和None（保持接口一致性）
        """
        # 特征映射
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器处理
        x = self.transformer_encoder(x)
        
        # 取序列最后一个时间步的特征进行预测
        x = x[:, -1, :]
        
        # 输出层预测
        out = self.output_layer(x)
        
        return out.squeeze(-1), None

def train_transformer_model(X_train, y_train, X_test, y_test, progress_callback=None):
    """
    训练Transformer模型
    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        X_test: 测试数据特征
        y_test: 测试数据标签
        progress_callback: 训练进度回调函数
    Returns:
        预测结果、真实值和训练好的模型
    """
    # 数据类型转换
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.FloatTensor(y_test)
    
    # 确保标签维度正确
    if len(y_train.shape) == 1:
        y_train = y_train.view(-1)
    if len(y_test.shape) == 1:
        y_test = y_test.view(-1)
    
    try:
        # 从配置文件加载训练参数
        batch_size = int(TRANSFORMER_PARAMS['batch_size'])
        epochs = int(TRANSFORMER_PARAMS['epochs'])
        learning_rate = float(TRANSFORMER_PARAMS['learning_rate'])
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型和训练组件
        model = Transformer()
        criterion = nn.MSELoss()  # 均方误差损失
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练循环
        best_model = None
        best_loss = float('inf')
        patience = 15  # 早停耐心值
        no_improve = 0  # 无改善轮数计数
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_test)
                val_loss = criterion(val_outputs, y_test)
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if progress_callback:
                progress_callback(epoch + 1, epochs)
        
        # 加载最佳模型
        model.load_state_dict(best_model)
        model.eval()
        
        # 生成预测结果
        with torch.no_grad():
            y_pred, _ = model(X_test)
            y_pred = y_pred.view(-1, 1)
        
        return y_pred.numpy(), y_test.view(-1, 1).numpy(), model
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e
