import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np
from models.transformer_model import Transformer, train_transformer_model
from config.config import TRANSFORMER_PARAMS
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # 创建测试数据目录
        self.test_dir = "./test_output"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
        # 生成模拟数据
        self.seq_length = 64
        self.feature_dim = TRANSFORMER_PARAMS['feature']
        self.batch_size = 32
        self.n_samples = 1000
        
        # 生成正弦波形数据作为测试数据
        t = np.linspace(0, 100, self.n_samples)
        self.data = np.zeros((self.n_samples, self.feature_dim))
        for i in range(self.feature_dim):
            self.data[:, i] = np.sin(t * (i + 1) / 10) + np.random.normal(0, 0.1, self.n_samples)
            
        # 归一化数据
        self.min_vals = np.min(self.data, axis=0)
        self.max_vals = np.max(self.data, axis=0)
        self.data = (self.data - self.min_vals) / (self.max_vals - self.min_vals)
        
        # 创建序列数据
        X, y = [], []
        for i in range(len(self.data) - self.seq_length):
            X.append(self.data[i:i+self.seq_length])
            y.append(self.data[i+self.seq_length, 1])  # 使用第二个特征作为目标
            
        # 转换为PyTorch张量
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(np.array(y))
        
        # 划分训练集和测试集
        train_size = int(0.8 * len(self.X))
        self.X_train = self.X[:train_size]
        self.y_train = self.y[:train_size]
        self.X_test = self.X[train_size:]
        self.y_test = self.y[train_size:]

    def test_model_initialization(self):
        """测试模型是否能正确初始化"""
        try:
            model = Transformer()
            self.assertIsNotNone(model)
            print("模型初始化测试通过")
        except Exception as e:
            self.fail(f"模型初始化失败: {str(e)}")

    def test_model_forward(self):
        """测试模型前向传播"""
        model = Transformer()
        try:
            batch_x = self.X[:self.batch_size]
            output, attention = model(batch_x)
            self.assertEqual(output.shape[0], self.batch_size)
            print("模型前向传播测试通过")
            print(f"输出形状: {output.shape}")
        except Exception as e:
            self.fail(f"前向传播失败: {str(e)}")

    def test_model_training(self):
        """测试模型训练过程"""
        try:
            # 定义进度回调函数
            def progress_callback(epoch, total_epochs):
                print(f"训练进度: {epoch}/{total_epochs}")

            # 训练模型
            y_pred, y_true, model = train_transformer_model(
                self.X_train, 
                self.y_train,
                self.X_test,
                self.y_test,
                progress_callback
            )
            
            # 检查预测结果
            self.assertEqual(y_pred.shape, y_true.shape)
            print("模型训练测试通过")
            print(f"预测形状: {y_pred.shape}")
            
            # 绘制预测结果
            self._plot_predictions(y_pred, y_true)
            
        except Exception as e:
            self.fail(f"模型训练失败: {str(e)}")

    def test_model_prediction(self):
        """测试模型预测功能"""
        model = Transformer()
        try:
            with torch.no_grad():
                test_batch = self.X_test[:self.batch_size]
                predictions, _ = model(test_batch)
                self.assertEqual(predictions.shape[0], self.batch_size)
                print("模型预测测试通过")
                print(f"预测形状: {predictions.shape}")
        except Exception as e:
            self.fail(f"模型预测失败: {str(e)}")

    def _plot_predictions(self, y_pred, y_true):
        """绘制预测结果对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.numpy()[:100], label='真实值', color='blue')
        plt.plot(y_pred.numpy()[:100], label='预测值', color='red', linestyle='--')
        plt.title('Transformer模型预测结果')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.test_dir, 'transformer_predictions.png'))
        plt.close()

    def tearDown(self):
        """清理测试环境"""
        # 如果需要清理测试输出目录，取消下面的注释
        # import shutil
        # shutil.rmtree(self.test_dir)
        pass

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 