import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from config.config import TRANSFORMER_PARAMS

# 从配置文件获取参数，并根据实际数据维度调整
d_model = 128  # 模型维度
d_ff = 256     # 前馈网络维度
d_k = d_v = 32  # K(=Q), V的维度
n_layers = 3    # Transformer层数
n_heads = 8     # 注意力头数
feature_dim = 15  # 输入特征维度，与数据匹配

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # 输入嵌入层，将特征维度映射到模型维度
        self.input_embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        
        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 取序列的最后一个时间步
        x = x[:, -1, :]
        
        # 输出层
        out = self.output_layer(x)
        
        return out.squeeze(-1), None  # 返回预测值和None（保持接口一致）

def train_transformer_model(X_train, y_train, X_test, y_test, progress_callback=None):
    # 确保输入数据是PyTorch张量
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.FloatTensor(y_test)
    
    # 确保维度正确
    if len(y_train.shape) == 1:
        y_train = y_train.view(-1)
    if len(y_test.shape) == 1:
        y_test = y_test.view(-1)
    
    try:
        # 从配置文件获取训练参数
        batch_size = int(TRANSFORMER_PARAMS['batch_size'])
        epochs = int(TRANSFORMER_PARAMS['epochs'])
        learning_rate = float(TRANSFORMER_PARAMS['learning_rate'])
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        model = Transformer()
        criterion = nn.MSELoss()  # 使用MSE损失函数
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        best_model = None
        best_loss = float('inf')
        patience = 15  # 早停耐心值
        no_improve = 0  # 没有改善的轮数
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 计算验证集损失
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
        
        # 预测
        with torch.no_grad():
            y_pred, _ = model(X_test)
            y_pred = y_pred.view(-1, 1)
        
        return y_pred.numpy(), y_test.view(-1, 1).numpy(), model
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e
