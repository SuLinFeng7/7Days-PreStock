"""
Transformer模型实现文件
基于Transformer架构的时间序列预测模型
主要特点：
1. 使用自注意力机制捕捉时间序列中的长期依赖关系
2. 包含位置编码以保留序列顺序信息
3. 多头注意力机制实现多角度特征提取
4. 添加残差连接和层归一化
5. 使用Huber损失提高对异常值的鲁棒性
6. 实现学习率预热和衰减策略
7. 添加梯度裁剪防止梯度爆炸
8. 支持动态参数调整
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from config.config import TRANSFORMER_PARAMS
import math

# 从配置文件加载模型参数
d_model = TRANSFORMER_PARAMS['d_model']
d_ff = TRANSFORMER_PARAMS['d_ff']
n_heads = TRANSFORMER_PARAMS['n_heads']
n_layers = TRANSFORMER_PARAMS['num_layers']
dropout = TRANSFORMER_PARAMS['dropout']
attention_dropout = TRANSFORMER_PARAMS['attention_dropout']
feature_dim = 15

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
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [seq_len, batch_size, d_model]
        Returns:
            添加位置编码后的序列
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """
    Transformer模型主体
    使用Transformer编码器架构进行时间序列预测
    """
    def __init__(self, params=None):
        super(TimeSeriesTransformer, self).__init__()
        
        # 使用传入的参数或默认参数
        if params is None:
            params = TRANSFORMER_PARAMS
        else:
            # 合并默认参数和传入参数
            default_params = TRANSFORMER_PARAMS.copy()
            default_params.update(params)
            params = default_params
        
        # 从参数中获取模型配置
        self.d_model = params.get('d_model', TRANSFORMER_PARAMS['d_model'])
        self.d_ff = params.get('d_ff', TRANSFORMER_PARAMS['d_ff'])
        self.n_heads = params.get('n_heads', TRANSFORMER_PARAMS['n_heads'])
        self.n_layers = params.get('num_layers', TRANSFORMER_PARAMS['num_layers'])
        self.dropout = params.get('dropout', TRANSFORMER_PARAMS['dropout'])
        self.attention_dropout = params.get('attention_dropout', TRANSFORMER_PARAMS['attention_dropout'])
        
        # 输入特征映射
        self.input_projection = nn.Linear(feature_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # 创建Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',  # 使用GELU激活函数
            batch_first=True,
            norm_first=True    # 使用Pre-LN结构
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # 输出预测层
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, src_mask=None):
        """
        前向传播函数
        Args:
            x: 输入数据 [batch_size, seq_len, feature_dim]
        Returns:
            预测值和None（保持接口一致性）
        """
        # 特征映射
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x, src_mask)
        
        # 取序列最后一个时间步的特征进行预测
        x = x[:, -1, :]
        
        # 输出层预测
        out = self.output_layer(x)
        
        return out.squeeze(-1), None

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.min(abs_diff, torch.tensor(self.delta))
        linear = abs_diff - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + self.delta * linear)

def create_attention_mask(seq_len):
    """创建因果注意力掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, learning_rate):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_mult = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate * lr_mult

def train_transformer_model(X_train, y_train, X_test, y_test, progress_callback=None, **kwargs):
    """
    训练Transformer模型
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
    try:
        # 数据类型转换
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # 获取训练参数
        batch_size = kwargs.get('batch_size', TRANSFORMER_PARAMS['batch_size'])
        epochs = kwargs.get('epochs', TRANSFORMER_PARAMS['epochs'])
        learning_rate = kwargs.get('learning_rate', TRANSFORMER_PARAMS['learning_rate'])
        weight_decay = kwargs.get('weight_decay', TRANSFORMER_PARAMS['weight_decay'])
        warmup_steps = kwargs.get('warmup_steps', TRANSFORMER_PARAMS['warmup_steps'])
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 初始化模型
        model = TimeSeriesTransformer(kwargs)
        print(f"\n模型参数总量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 使用Huber损失
        criterion = HuberLoss(delta=1.0)
        
        # 使用AdamW优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 创建学习率调度器
        total_steps = len(train_loader) * epochs
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            learning_rate=learning_rate
        )
        
        # 创建注意力掩码
        src_mask = create_attention_mask(X_train.shape[1])
        
        # 训练循环
        best_model = None
        best_loss = float('inf')
        patience = kwargs.get('patience', 15)
        no_improve = 0
        
        print("\n开始训练循环...")
        print(f"总训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"注意力头数: {kwargs.get('n_heads', TRANSFORMER_PARAMS['n_heads'])}")
        print(f"Dropout率: {kwargs.get('dropout', TRANSFORMER_PARAMS['dropout'])}")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs, _ = model(batch_X, src_mask)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # 计算平均损失
            avg_loss = total_loss / batch_count
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_test, src_mask)
                val_loss = criterion(val_outputs, y_test)
            
            # 更新进度
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss, val_loss.item())
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n早停：验证损失在{patience}轮内没有改善")
                    break
        
        # 加载最佳模型
        if best_model is not None:
            model.load_state_dict(best_model)
        
        # 生成预测结果
        model.eval()
        with torch.no_grad():
            y_pred, _ = model(X_test, src_mask)
        
        return y_pred.numpy().reshape(-1, 1), y_test.numpy().reshape(-1, 1), model
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e
