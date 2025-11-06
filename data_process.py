# -*- coding:utf-8 -*-

import sys
import numpy as np
import pandas as pd
import torch
from args import args_parser
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import geopandas as gpd
from shapely.geometry import Point
import os

sys.path.append('../')
from torch.utils.data import Dataset, DataLoader

args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]

# 地理特征配置（模拟不同风电场的地理位置）
GEO_FEATURES = {
    'Task1_W_Zone1': {'latitude': 40.5, 'longitude': 116.4, 'altitude': 50, 'terrain': 0.8, 'coastal_distance': 150},
    'Task1_W_Zone2': {'latitude': 39.9, 'longitude': 116.2, 'altitude': 45, 'terrain': 0.6, 'coastal_distance': 180},
    'Task1_W_Zone3': {'latitude': 40.2, 'longitude': 116.6, 'altitude': 60, 'terrain': 0.7, 'coastal_distance': 120},
    'Task1_W_Zone4': {'latitude': 39.8, 'longitude': 116.3, 'altitude': 55, 'terrain': 0.5, 'coastal_distance': 200},
    'Task1_W_Zone5': {'latitude': 40.1, 'longitude': 116.5, 'altitude': 48, 'terrain': 0.9, 'coastal_distance': 100},
    'Task1_W_Zone6': {'latitude': 40.3, 'longitude': 116.1, 'altitude': 65, 'terrain': 0.4, 'coastal_distance': 250},
    'Task1_W_Zone7': {'latitude': 39.7, 'longitude': 116.7, 'altitude': 52, 'terrain': 0.3, 'coastal_distance': 300},
    'Task1_W_Zone8': {'latitude': 40.4, 'longitude': 116.8, 'altitude': 58, 'terrain': 0.2, 'coastal_distance': 80},
    'Task1_W_Zone9': {'latitude': 39.6, 'longitude': 116.9, 'altitude': 62, 'terrain': 0.1, 'coastal_distance': 350},
    'Task1_W_Zone10': {'latitude': 40.0, 'longitude': 116.0, 'altitude': 47, 'terrain': 0.85, 'coastal_distance': 90}
}


def load_data(file_name):
    """加载并预处理数据，包含气象和地理特征"""
    try:
        df = pd.read_csv('data/Wind/TaskDataSet 1/Task1_W_Zone1_10/' + file_name + '.csv', encoding='gbk')
    except FileNotFoundError:
        # 如果文件不存在，生成模拟数据用于测试
        print(f"File {file_name} not found, generating simulated data...")
        df = generate_simulated_data(file_name)

    columns = df.columns
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 标准化数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # 添加地理特征
    if file_name in GEO_FEATURES:
        geo_data = GEO_FEATURES[file_name]
        df['latitude'] = geo_data['latitude']
        df['longitude'] = geo_data['longitude']
        df['altitude'] = geo_data['altitude']
        df['terrain_roughness'] = geo_data['terrain']
        df['coastal_distance'] = geo_data['coastal_distance']

    return df


def generate_simulated_data(file_name, n_samples=1000):
    """生成模拟的风力涡轮机数据用于测试"""
    np.random.seed(hash(file_name) % 10000)

    # 基础时间序列
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')

    # 模拟气象特征
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 2, n_samples)
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 12) + np.random.normal(0, 5, n_samples)
    pressure = 1013 + 10 * np.cos(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1, n_samples)

    # 风速数据（U10, V10, U100, V100）
    u10 = 5 + 3 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1, n_samples)
    v10 = 3 + 2 * np.cos(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 0.5, n_samples)
    u100 = 7 + 4 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1.5, n_samples)
    v100 = 4 + 3 * np.cos(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1, n_samples)

    # 目标值（功率预测）
    wind_speed_effective = np.sqrt(u100 ** 2 + v100 ** 2)
    power_output = 0.3 * wind_speed_effective ** 3 + np.random.normal(0, 0.1, n_samples)

    data = {
        'Timestamp': time_index,
        'Temperature': temperature,
        'Humidity': humidity,
        'Pressure': pressure,
        'U10': u10,
        'V10': v10,
        'U100': u100,
        'V100': v100,
        'TargetVal': power_output
    }

    return pd.DataFrame(data)


def extract_geographic_features(file_name):
    """提取地理特征用于个性化模型"""
    if file_name in GEO_FEATURES:
        geo_data = GEO_FEATURES[file_name]
        features = [
            geo_data['latitude'],
            geo_data['longitude'],
            geo_data['altitude'],
            geo_data['terrain'],
            geo_data['coastal_distance']
        ]
        return torch.FloatTensor(features)
    else:
        # 默认地理特征
        return torch.FloatTensor([40.0, 116.0, 50.0, 0.5, 150.0])


class WindTurbineDataset(Dataset):
    """风力涡轮机数据集，包含气象和地理特征"""

    def __init__(self, data, geographic_features=None):
        self.data = data
        self.geographic_features = geographic_features

    def __getitem__(self, item):
        if self.geographic_features is not None:
            return self.data[item][0], self.data[item][1], self.geographic_features
        else:
            return self.data[item][0], self.data[item][1]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B):
    """处理风力数据序列，返回数据加载器"""
    data = load_data(file_name)
    geographic_features = extract_geographic_features(file_name)

    columns = data.columns
    wind = data[columns[2]] if len(columns) > 2 else data['U10']
    wind = wind.tolist()
    data = data.values.tolist()

    X, Y = [], []
    seq = []

    for i in range(len(data) - 30):
        train_seq = []
        train_label = []

        # 时间序列特征（风速历史）
        for j in range(i, i + args.seq_len):
            train_seq.append(wind[j])

        # 气象特征
        for c in range(3, min(7, len(data[0]))):  # U10, V10, U100, V100
            train_seq.append(data[i + args.seq_len][c])

        # 添加额外气象特征（如果存在）
        if len(data[0]) > 7:
            for c in range(7, min(10, len(data[0]))):  # 温度、湿度、气压等
                train_seq.append(data[i + args.seq_len][c])

        train_label.append(wind[i + args.seq_len])

        train_seq = torch.FloatTensor(train_seq).view(-1)
        train_label = torch.FloatTensor(train_label).view(-1)

        seq.append((train_seq, train_label))

    # 数据分割
    train_size = int(len(seq) * args.train_ratio)
    val_size = int(len(seq) * args.val_ratio)

    Dtr = seq[0:train_size]
    Dval = seq[train_size:train_size + val_size]
    Dte = seq[train_size + val_size:]

    # 调整长度以适应批次大小
    train_len = int(len(Dtr) / B) * B
    val_len = int(len(Dval) / B) * B
    test_len = int(len(Dte) / B) * B

    Dtr, Dval, Dte = Dtr[:train_len], Dval[:val_len], Dte[:test_len]

    # 创建数据加载器
    train_dataset = WindTurbineDataset(Dtr, geographic_features)
    val_dataset = WindTurbineDataset(Dval, geographic_features)
    test_dataset = WindTurbineDataset(Dte, geographic_features)

    Dtr = DataLoader(dataset=train_dataset, batch_size=B, shuffle=True, num_workers=0)
    Dval = DataLoader(dataset=val_dataset, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test_dataset, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dval, Dte


def calculate_data_quality_metrics(file_name, data_loader):
    """计算数据质量指标用于信用机制"""
    if len(data_loader) == 0:
        return 0.0

    all_data = []
    all_targets = []

    for batch in data_loader:
        if len(batch) == 3:  # 包含地理特征
            sequences, targets, _ = batch
        else:
            sequences, targets = batch

        all_data.extend(sequences.numpy().flatten())
        all_targets.extend(targets.numpy().flatten())

    if len(all_data) == 0:
        return 0.0

    # 计算数据质量指标
    data_variance = np.var(all_data)
    target_variance = np.var(all_targets)
    data_volume = len(all_data)

    # 综合质量评分
    quality_score = min(1.0,
                        (0.3 * min(1.0, data_variance) +
                         0.3 * min(1.0, target_variance) +
                         0.4 * min(1.0, data_volume / 1000))
                        )

    return quality_score


def get_mape(x, y):
    """计算平均绝对百分比误差"""
    return np.mean(np.abs((x - y) / np.clip(x, 1e-8, None)))


def get_rmse(x, y):
    """计算均方根误差"""
    return np.sqrt(np.mean((x - y) ** 2))


def get_mae(x, y):
    """计算平均绝对误差"""
    return np.mean(np.abs(x - y))