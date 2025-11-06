# -*- coding:utf-8 -*-

import torch
import numpy as np
from client import train, test
from model import HybridModel
import copy
from torch.utils.tensorboard import SummaryWriter

import tensorboard as tb
import tensorflow as tf

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

writer = SummaryWriter('run/2')


class FedPer:
    def __init__(self, args):
        self.args = args
        self.nn = HybridModel(args=self.args, name='server').to(args.device)
        self.nns = []
        self.client_credits = np.ones(self.args.K)
        self.client_data_quality = np.ones(self.args.K)

        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.r):
            print('round', t + 1, ':')

            self.evaluate_data_quality()
            selected_clients = self.select_clients_by_credit()

            if len(selected_clients) > 0:
                self.dispatch(selected_clients)
                self.client_update(selected_clients)
                self.aggregation(selected_clients)
                self.update_client_credits(selected_clients)

        return self.nn

    def evaluate_data_quality(self):
        for j in range(self.args.K):
            client_model = self.nns[j]
            quality_score = self.calculate_data_quality(client_model)
            self.client_data_quality[j] = quality_score

    def calculate_data_quality(self, model):
        model.eval()
        quality_metrics = 0.0

        if hasattr(model, 'get_quality_metrics'):
            quality_metrics = model.get_quality_metrics()
        else:
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        quality_metrics += torch.norm(param.grad).item()

        return max(0.1, min(1.0, quality_metrics))

    def select_clients_by_credit(self):
        selected_clients = []
        credit_threshold = self.args.credit_threshold

        for j in range(self.args.K):
            credit_score = self.client_credits[j] * self.client_data_quality[j]
            if credit_score >= credit_threshold:
                selected_clients.append(j)

        print(
            f"Selected clients: {selected_clients} with credits: {[self.client_credits[i] for i in selected_clients]}")
        return selected_clients

    def update_client_credits(self, selected_clients):
        for j in selected_clients:
            client_model = self.nns[j]

            performance_improvement = self.assess_performance_improvement(j)
            data_consistency = self.assess_data_consistency(j)

            credit_update = 0.8 * performance_improvement + 0.2 * data_consistency
            self.client_credits[j] = 0.9 * self.client_credits[j] + 0.1 * credit_update
            self.client_credits[j] = max(0.1, min(1.0, self.client_credits[j]))

    def assess_performance_improvement(self, client_id):
        return np.random.uniform(0.7, 1.0)

    def assess_data_consistency(self, client_id):
        return np.random.uniform(0.8, 1.0)

    def aggregation(self, selected_clients):
        if len(selected_clients) == 0:
            return

        total_weight = 0
        for j in selected_clients:
            total_weight += self.nns[j].len * self.client_credits[j]

        for v in self.nn.base_layers.parameters():
            v.data.zero_()

        for j in selected_clients:
            client_weight = (self.nns[j].len * self.client_credits[j]) / total_weight
            cnt = 0

            for server_param, client_param in zip(self.nn.base_layers.parameters(),
                                                  self.nns[j].base_layers.parameters()):
                server_param.data += client_param.data * client_weight
                cnt += 1

    def dispatch(self, selected_clients):
        for j in selected_clients:
            cnt = 0
            for server_param, client_param in zip(self.nn.base_layers.parameters(),
                                                  self.nns[j].base_layers.parameters()):
                client_param.data = server_param.data.clone()
                cnt += 1

    def client_update(self, selected_clients):
        for k in selected_clients:
            self.nns[k] = train(self.args, self.nns[k])

    def global_test(self):
        model_performances = {}

        for j in range(self.args.K):
            model = self.nns[j]
            model.eval()
            performance = test(self.args, model)
            model_performances[j] = performance

            print(f"Client {j} - Credit: {self.client_credits[j]:.3f}, Performance: {performance}")

        return model_performances

    def get_personalized_models(self):
        return self.nns

    def get_client_credits(self):
        return self.client_credits