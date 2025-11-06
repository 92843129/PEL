# -*- coding:utf-8 -*-

from args import args_parser
from server import FedPer
import torch
import numpy as np
import json
import os
from datetime import datetime


def setup_experiment(args):
    """设置实验环境和保存路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{args.experiment_name}_{timestamp}"

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/results", exist_ok=True)

    # 保存实验参数
    with open(f"{experiment_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    return experiment_dir


def save_results(experiment_dir, results, client_credits, model_performances):
    """保存实验结果"""
    # 保存整体结果
    with open(f"{experiment_dir}/results/overall_results.json", 'w') as f:
        json.dump(results, f, indent=4)

    # 保存客户端信用分
    credit_results = {
        'client_credits': client_credits.tolist() if isinstance(client_credits, np.ndarray) else client_credits,
        'credit_evolution': results.get('credit_evolution', [])
    }
    with open(f"{experiment_dir}/results/client_credits.json", 'w') as f:
        json.dump(credit_results, f, indent=4)

    # 保存模型性能
    with open(f"{experiment_dir}/results/model_performances.json", 'w') as f:
        json.dump(model_performances, f, indent=4)



def main():
    # 解析参数
    args = args_parser()

    # 设置设备
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {args.device}")

    # 设置实验环境
    experiment_dir = setup_experiment(args)
    print(f"Experiment directory: {experiment_dir}")

    try:
        # 初始化联邦学习系统
        print("Initializing Personalized Federated Learning System...")
        fedPer = FedPer(args)

        # 记录信用分演化
        credit_evolution = []

        # 运行联邦学习训练
        print("\nStarting Federated Learning Training...")
        global_model = fedPer.server()

        # 记录最终信用分
        credit_evolution.append(fedPer.client_credits.copy())

        # 保存全局模型
        torch.save(global_model.state_dict(), f"{experiment_dir}/models/global_model.pth")

        # 保存个性化模型
        personalized_models = fedPer.get_personalized_models()
        for i, model in enumerate(personalized_models):
            torch.save(model.state_dict(), f"{experiment_dir}/models/personalized_model_client_{i}.pth")

        # 全局测试
        print("\nStarting Global Testing...")
        model_performances = fedPer.global_test()

        # 收集结果
        results = {
            'experiment_name': args.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'communication_rounds': args.r,
            'total_clients': args.K,
            'credit_evolution': credit_evolution,
            'final_credits': fedPer.client_credits.tolist() if isinstance(fedPer.client_credits,
                                                                          np.ndarray) else fedPer.client_credits
        }

        # 保存结果
        save_results(experiment_dir, results, fedPer.client_credits, model_performances)


        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {experiment_dir}")

    except Exception as e:
        print(f"Error during experiment: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()