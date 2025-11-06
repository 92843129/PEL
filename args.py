# -*- coding:utf-8 -*-

import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser(description='Personalized Federated Learning for Wind Turbine Power Prediction')

    # 基础联邦学习参数
    parser.add_argument('--E', type=int, default=10, help='number of local epochs')  # 本地训练轮数
    parser.add_argument('--r', type=int, default=60, help='number of communication rounds')  # 通信轮数
    parser.add_argument('--K', type=int, default=10, help='number of total clients')  # 客户端数量
    parser.add_argument('--input_dim', type=int, default=28, help='input dimension')  # 输入维度
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension')  # 输出维度
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')  # 基础学习率
    parser.add_argument('--C', type=float, default=0.7, help='client sampling rate')  # 客户端采样率
    parser.add_argument('--B', type=int, default=50, help='local batch size')  # 本地批次大小

    # 个性化联邦学习参数
    parser.add_argument('--Kp', type=int, default=2, help='number of personalized layers')  # 个性化层数量
    parser.add_argument('--total', type=int, default=6, help='number of total layers')  # 总层数
    parser.add_argument('--personal_lr_multiplier', type=float, default=1.5,
                        help='learning rate multiplier for personal layers')  # 个性化层学习率倍数
    parser.add_argument('--lambda_reg', type=float, default=0.01,
                        help='regularization coefficient for personalization')  # 个性化正则化系数

    # 信用感知参与机制参数
    parser.add_argument('--credit_threshold', type=float, default=0.6,
                        help='minimum credit score for client participation')  # 客户端参与的最低信用分
    parser.add_argument('--credit_decay', type=float, default=0.9, help='credit decay factor')  # 信用衰减因子
    parser.add_argument('--quality_weight', type=float, default=0.8,
                        help='weight for data quality in credit calculation')  # 数据质量在信用计算中的权重
    parser.add_argument('--performance_weight', type=float, default=0.2,
                        help='weight for performance in credit calculation')  # 性能在信用计算中的权重

    # 混合模型架构参数
    parser.add_argument('--base_hidden_dim', type=int, default=64, help='hidden dimension for base layers')  # 基础层隐藏维度
    parser.add_argument('--personal_hidden_dim', type=int, default=32,
                        help='hidden dimension for personal layers')  # 个性化层隐藏维度
    parser.add_argument('--meteo_features', type=int, default=8, help='number of meteorological features')  # 气象特征数量
    parser.add_argument('--turbine_features', type=int, default=6,
                        help='number of turbine-specific features')  # 风机特定特征数量
    parser.add_argument('--geo_features', type=int, default=4, help='number of geographic features')  # 地理特征数量

    # 训练优化参数
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'],
                        help='type of optimizer')  # 优化器类型
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')  # 权重衰减
    parser.add_argument('--step_size', type=int, default=15, help='learning rate decay step size')  # 学习率衰减步长
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay gamma')  # 学习率衰减系数
    parser.add_argument('--gradient_clip', type=bool, default=True, help='whether to use gradient clipping')  # 是否使用梯度裁剪
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clipping value')  # 梯度裁剪值

    # 设备与实验设置
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='computing device')  # 计算设备
    parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available(), help='whether to use GPU')  # 是否使用GPU
    parser.add_argument('--experiment_name', type=str, default='wind_turbine_pfl',
                        help='experiment name for logging')  # 实验名称
    parser.add_argument('--save_interval', type=int, default=10, help='model saving interval')  # 模型保存间隔

    # 数据相关参数
    parser.add_argument('--seq_len', type=int, default=24, help='sequence length for time series data')  # 时间序列数据长度
    parser.add_argument('--pred_len', type=int, default=1, help='prediction length')  # 预测长度
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training data ratio')  # 训练数据比例
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation data ratio')  # 验证数据比例
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test data ratio')  # 测试数据比例

    # 客户端配置
    clients = ['Task1_W_Zone' + str(i) for i in range(1, 11)]  # 客户端名称
    parser.add_argument('--clients', default=clients, help='client names')  # 客户端列表

    # 地理特征配置
    parser.add_argument('--geo_adaptation', type=bool, default=True,
                        help='whether to use geographic adaptation')  # 是否使用地理适应
    parser.add_argument('--geo_weight', type=float, default=0.1,
                        help='weight for geographic adaptation loss')  # 地理适应损失权重

    args = parser.parse_args()  # 解析参数

    # 参数验证
    if args.Kp >= args.total:
        raise ValueError("Number of personalized layers must be less than total layers")

    if args.credit_threshold < 0 or args.credit_threshold > 1:
        raise ValueError("Credit threshold must be between 0 and 1")

    print("=" * 60)
    print("Personalized Federated Learning Configuration")
    print("=" * 60)
    print(f"Total Clients: {args.K}")
    print(f"Communication Rounds: {args.r}")
    print(f"Personalized Layers: {args.Kp}/{args.total}")
    print(f"Credit Threshold: {args.credit_threshold}")
    print(f"Geographic Adaptation: {args.geo_adaptation}")
    print(f"Device: {args.device}")
    print("=" * 60)

    return args