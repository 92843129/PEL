# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class HybridModel(nn.Module):
    """混合联邦学习模型：结合本地风机模型和全局大气模型"""

    def __init__(self, args, name):
        super(HybridModel, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.args = args

        # 基础层（共享层）- 处理通用特征
        self.base_layers = self._build_base_layers()

        # 个性化层 - 适应地理特征
        self.personal_layers = self._build_personal_layers()

        # 气象特征处理层
        self.meteo_layers = self._build_meteo_layers()

        # 地理特征嵌入
        self.geo_embedding = nn.Linear(args.geo_features, 8)

        # 特征融合层
        self.fusion_layer = nn.Linear(32 + 16 + 8, 64)
        self.output_layer = nn.Linear(64, args.output_dim)

        # 数据质量指标
        self.data_quality = 1.0
        self.performance_history = []

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def _build_base_layers(self):
        """构建基础共享层"""
        base_layers = nn.Sequential(
            nn.Linear(self.args.input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        return base_layers

    def _build_personal_layers(self):
        """构建个性化层"""
        personal_layers = nn.Sequential(
            nn.Linear(32, 16),  # 输入来自基础层
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 16),
            nn.Tanh()
        )
        return personal_layers

    def _build_meteo_layers(self):
        """构建气象特征处理层"""
        meteo_layers = nn.Sequential(
            nn.Linear(self.args.meteo_features, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        return meteo_layers

    def forward(self, data):
        """前向传播"""
        if len(data) == 3:
            # 包含地理特征的情况
            sequence, meteo_features, geo_features = data
            geo_embedded = self.geo_embedding(geo_features)
        else:
            # 只有序列数据的情况
            sequence = data
            meteo_features = sequence[:, -self.args.meteo_features:]
            geo_features = torch.zeros(sequence.size(0), self.args.geo_features).to(sequence.device)
            geo_embedded = self.geo_embedding(geo_features)

        # 基础层处理
        base_features = self.base_layers(sequence)

        # 个性化层处理
        personal_features = self.personal_layers(base_features)

        # 气象特征处理
        meteo_processed = self.meteo_layers(meteo_features)

        # 特征融合
        combined_features = torch.cat([base_features, personal_features, meteo_processed, geo_embedded], dim=1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.relu(fused_features)
        fused_features = self.dropout(fused_features)

        # 输出层
        output = self.output_layer(fused_features)

        return output

    def extract_geographic_features(self):
        """提取地理特征用于个性化适应"""
        # 这里可以集成真实的地理数据
        # 目前返回模拟的地理特征向量
        if self.name in ['Task1_W_Zone' + str(i) for i in range(1, 11)]:
            zone_id = int(self.name.split('Zone')[-1])
            geo_features = torch.FloatTensor([
                (zone_id - 1) * 0.1,  # 纬度变化
                (zone_id - 1) * 0.05,  # 经度变化
                (zone_id - 1) * 2,  # 海拔变化
                (zone_id - 1) * 0.08  # 地形变化
            ])
        else:
            geo_features = torch.FloatTensor([0.5, 0.5, 50.0, 0.5])

        return geo_features

    def adapt_to_geography(self, geographic_features):
        """根据地理特征调整个性化层"""
        # 这里可以实现根据地理特征动态调整个性化层参数
        # 目前使用简单的权重调整
        with torch.no_grad():
            for param in self.personal_layers.parameters():
                adjustment = geographic_features.mean().item() * 0.01
                param.data *= (1.0 + adjustment)

    def adapt_personal_layers(self, geographic_features):
        """调整个性化层以适应特定地理特征"""
        # 更精细的个性化层调整
        geo_weight = geographic_features.abs().mean().item()

        with torch.no_grad():
            for name, param in self.personal_layers.named_parameters():
                if 'weight' in name:
                    # 根据地理特征调整权重
                    adjustment = torch.sigmoid(torch.tensor(geo_weight * 0.1 - 0.05))
                    param.data = param.data * (0.9 + 0.2 * adjustment)

    def personal_regularization(self):
        """个性化正则化项，防止个性化层过度偏离"""
        reg_loss = 0.0
        for param in self.personal_layers.parameters():
            reg_loss += torch.norm(param, p=2)
        return reg_loss

    def update_data_quality_metrics(self, train_loader, test_loader):
        """更新数据质量指标"""
        self.eval()
        train_predictions = []
        train_targets = []

        with torch.no_grad():
            for batch in train_loader:
                if len(batch) == 3:
                    sequences, targets, _ = batch
                else:
                    sequences, targets = batch

                outputs = self(sequences)
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())

        # 计算训练数据的预测一致性
        if len(train_predictions) > 1:
            predictions_array = np.array(train_predictions)
            targets_array = np.array(train_targets)

            mae = np.mean(np.abs(predictions_array - targets_array))
            consistency = 1.0 / (1.0 + mae)

            self.data_quality = min(1.0, max(0.1, consistency))

    def get_quality_metrics(self):
        """获取数据质量指标"""
        return self.data_quality

    def evaluate_geographic_adaptation(self):
        """评估地理适应性能"""
        # 模拟地理适应评分
        adaptation_score = 0.8 + 0.2 * torch.sigmoid(torch.tensor(self.data_quality - 0.5))
        return adaptation_score.item()

    def freeze_base_layers(self):
        """冻结基础层参数"""
        for param in self.base_layers.parameters():
            param.requires_grad = False

    def unfreeze_base_layers(self):
        """解冻基础层参数"""
        for param in self.base_layers.parameters():
            param.requires_grad = True

    def freeze_personal_layers(self):
        """冻结个性化层参数"""
        for param in self.personal_layers.parameters():
            param.requires_grad = False

    def unfreeze_personal_layers(self):
        """解冻个性化层参数"""
        for param in self.personal_layers.parameters():
            param.requires_grad = True


class ANN(nn.Module):
    """传统ANN模型（用于对比）"""

    def __init__(self, args, name):
        super(ANN, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.fc1 = nn.Linear(args.input_dim, 16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x