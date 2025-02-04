import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import tushare as ts
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from config.config import TUSHARE_TOKEN, TRAIN_TEST_SPLIT, TRAIN_YEARS

def get_stock_data(stock_id, start_date, end_date):
    """获取股票数据"""
    try:
        # 判断是否为A股
        if '.SH' in stock_id or '.SZ' in stock_id:
            # 使用tushare获取A股数据
            pro = ts.pro_api(TUSHARE_TOKEN)
            
            # 先尝试获取股票的所有历史数据
            # 使用一个较早的日期作为起始日期，确保能获取到所有数据
            start_date_ts = '20100101'  # 使用2010年作为起始日期
            end_date_ts = end_date.replace('-', '')
            
            df = pro.daily(ts_code=stock_id, 
                          start_date=start_date_ts, 
                          end_date=end_date_ts)
            
            if df.empty:
                raise ValueError(f"No data found for stock {stock_id}")
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df.sort_index(inplace=True)
            
            stock_data = df[['close']]
            
        else:
            # 使用yfinance获取其他市场数据
            # 先尝试获取所有历史数据
            stock = yf.download(stock_id, end=end_date, progress=False)
            
            if stock.empty:
                raise ValueError(f"No data found for ticker {stock_id}")
            
            if 'Adj Close' in stock.columns:
                stock_data = stock[['Adj Close']].rename(columns={"Adj Close": "close"})
            else:
                stock_data = stock[['Close']].rename(columns={"Close": "close"})
        
        # 获取第一条数据的日期
        first_date = stock_data.index[0]
        target_start_date = datetime.now() - timedelta(days=365*TRAIN_YEARS)
        
        # 打印第一条数据的日期
        print(f"\n股票 {stock_id} 的第一条数据日期为: {first_date.strftime('%Y-%m-%d')}")
        print(f"目标起始日期为: {target_start_date.strftime('%Y-%m-%d')}")
        
        # 如果第一条数据的日期晚于目标起始日期，使用所有可用数据
        # 否则，使用最近TRAIN_YEARS年的数据
        if first_date > target_start_date:
            print(f"股票上市时间不足{TRAIN_YEARS}年，将使用所有可用数据")
            final_data = stock_data
        else:
            print(f"股票上市时间超过{TRAIN_YEARS}年，将使用最近{TRAIN_YEARS}年的数据")
            final_data = stock_data[stock_data.index >= target_start_date]
        
        print(f"最终使用的数据范围: {final_data.index[0].strftime('%Y-%m-%d')} 到 {final_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"数据总量: {len(final_data)} 条")
        
        return final_data
        
    except Exception as e:
        raise Exception(f"Error fetching data for {stock_id}: {str(e)}")

def create_sequences(data, window_size, prediction_steps):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(window_size, len(data) - prediction_steps):
        X.append(data[i - window_size:i, 0])
        y.append(data[i + prediction_steps - 1, 0])
    return np.array(X), np.array(y)

def prepare_prediction_data(df, window_size=20):
    """准备预测数据
    使用数据总量的4/5作为训练集，1/5作为测试集
    """
    if 'adjClose' in df.columns:
        data = df['adjClose'].values.reshape(-1, 1)
    else:
        data = df['close'].values.reshape(-1, 1)
    
    # 添加技术指标
    df_temp = pd.DataFrame(data, columns=['close'], index=df.index)  # 保留原始索引
    
    # 基础技术指标
    df_temp['MA5'] = df_temp['close'].rolling(window=5).mean()
    df_temp['MA20'] = df_temp['close'].rolling(window=20).mean()
    df_temp['MA60'] = df_temp['close'].rolling(window=60).mean()
    df_temp['RSI'] = calculate_rsi(df_temp['close'])
    df_temp['MACD'], df_temp['Signal'] = calculate_macd(df_temp['close'])
    
    # 添加波动率指标
    df_temp['Volatility'] = df_temp['close'].pct_change().rolling(window=20).std()
    
    # 添加动量指标
    df_temp['ROC'] = calculate_roc(df_temp['close'])  # 变动率
    df_temp['MOM'] = calculate_momentum(df_temp['close'])  # 动量
    
    # 添加布林带
    df_temp['Upper_BB'], df_temp['Lower_BB'] = calculate_bollinger_bands(df_temp['close'])
    
    # 添加KDJ指标
    df_temp['K'], df_temp['D'], df_temp['J'] = calculate_kdj(df_temp['close'])
    
    # 删除NaN值
    df_temp = df_temp.dropna()
    
    # 特征选择
    features = df_temp[['close', 'MA5', 'MA20', 'MA60', 'RSI', 'MACD', 'Signal',
                       'Volatility', 'ROC', 'MOM', 'Upper_BB', 'Lower_BB',
                       'K', 'D', 'J']].values
    
    # 归一化处理
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_price = price_scaler.fit_transform(df_temp[['close']].values)
    scaled_features = feature_scaler.fit_transform(features)
    
    # 创建序列数据
    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i-window_size:i])
        y.append(scaled_price[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # 修改训练测试集划分逻辑
    total_samples = len(X)
    train_size = int(total_samples * TRAIN_TEST_SPLIT)  # 使用4/5的数据作为训练集
    
    # 划分训练集和测试集
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # 获取对应的日期范围
    train_start_date = df_temp.index[window_size]
    train_end_date = df_temp.index[train_size+window_size-1]
    test_start_date = df_temp.index[train_size+window_size]
    test_end_date = df_temp.index[-1]
    
    print(f"\n数据集划分情况:")
    print(f"训练集大小: {len(X_train)} 样本")
    print(f"测试集大小: {len(X_test)} 样本")
    print(f"训练集时间范围: {train_start_date.strftime('%Y-%m-%d')} 到 {train_end_date.strftime('%Y-%m-%d')}")
    print(f"测试集时间范围: {test_start_date.strftime('%Y-%m-%d')} 到 {test_end_date.strftime('%Y-%m-%d')}")
    
    return X_train, y_train, X_test, y_test, price_scaler, feature_scaler

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_roc(prices, period=12):
    """计算变动率"""
    return prices.pct_change(period) * 100

def calculate_momentum(prices, period=10):
    """计算动量"""
    return prices.diff(period)

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """计算布林带"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_kdj(prices, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    low_list = prices.rolling(window=n).min()
    high_list = prices.rolling(window=n).max()
    rsv = (prices - low_list) / (high_list - low_list) * 100
    K = rsv.ewm(com=m1-1, adjust=False).mean()
    D = K.ewm(com=m2-1, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J
