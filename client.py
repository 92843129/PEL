# -*- coding:utf-8 -*-

from itertools import chain
import sys
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os

from data_process import nn_seq_wind
import pandas as pd
import csv, codecs

from torch.utils.tensorboard import SummaryWriter

import tensorboard as tb
import tensorflow as tf

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

writer = SummaryWriter('run/1')


def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer_csv = csv.writer(file_csv)
    for data in datas:
        writer_csv.writerow(data)
    print("保存文件成功，处理结束")


def train(args, model):
    model.train()
    Dtr, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)

    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.base_layers.parameters(), 'lr': lr},
            {'params': model.personal_layers.parameters(), 'lr': lr * args.personal_lr_multiplier}
        ], lr=lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.base_layers.parameters(), 'lr': lr},
            {'params': model.personal_layers.parameters(), 'lr': lr * args.personal_lr_multiplier}
        ], lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(f'training client {model.name}...')
    loss_function = nn.MSELoss().to(args.device)

    geographic_features = model.extract_geographic_features()
    model.adapt_to_geography(geographic_features)

    loss = 0
    loss_data = []

    for epoch in range(args.E):
        epoch_loss = 0
        num_batches = 0

        for (seq, label) in Dtr:
            seq = seq.to(args.device)
            label = label.to(args.device)

            y_pred = model(seq)
            loss = loss_function(y_pred, label)

            if hasattr(model, 'personal_regularization'):
                personal_reg = model.personal_regularization()
                loss = loss + args.lambda_reg * personal_reg

            loss_data.append([loss.item()])
            epoch_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()

            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

            optimizer.step()

        lr_step.step()

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f'Client {model.name} - Epoch {epoch}, Loss: {avg_epoch_loss:.6f}')

        writer.add_scalar(f'training_loss/client_{model.name}', avg_epoch_loss, epoch)

    model.update_data_quality_metrics(Dtr, Dte)

    return model


def test(args, ann):
    ann.eval()
    Dtr, Dte = nn_seq_wind(ann.name, args.B)
    pred = []
    y = []

    geographic_performance = {}

    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))

    pred = np.array(pred)
    y = np.array(y)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    print(f'Client {ann.name} - mae: {mae:.4f}, rmse: {rmse:.4f}')

    if hasattr(ann, 'evaluate_geographic_adaptation'):
        geo_adapt_score = ann.evaluate_geographic_adaptation()
        print(f'Client {ann.name} - Geographic Adaptation Score: {geo_adapt_score:.4f}')

    performance_metrics = {
        'mae': mae,
        'rmse': rmse,
        'geographic_adaptation': geo_adapt_score if 'geo_adapt_score' in locals() else 0.0
    }

    writer.add_scalar(f'mae/client_{ann.name}', mae, 0)
    writer.add_scalar(f'rmse/client_{ann.name}', rmse, 0)

    return performance_metrics


def evaluate_client_data_quality(args, model):
    model.eval()
    Dtr, Dte = nn_seq_wind(model.name, args.B)

    quality_metrics = {
        'data_volume': len(Dtr),
        'data_variance': 0.0,
        'prediction_consistency': 0.0,
        'noise_level': 0.0
    }

    if len(Dtr) > 0:
        all_data = []
        for (seq, label) in Dtr:
            all_data.extend(seq.flatten().tolist())

        if len(all_data) > 1:
            quality_metrics['data_variance'] = np.var(all_data)

    pred_consistency = evaluate_prediction_consistency(model, Dte)
    quality_metrics['prediction_consistency'] = pred_consistency

    noise_level = estimate_noise_level(model, Dtr)
    quality_metrics['noise_level'] = noise_level

    overall_quality = calculate_overall_quality_score(quality_metrics)
    quality_metrics['overall_quality'] = overall_quality

    return quality_metrics


def evaluate_prediction_consistency(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for (seq, label) in data_loader:
            seq = seq.to(next(model.parameters()).device)
            pred = model(seq)
            predictions.extend(pred.cpu().numpy())

    if len(predictions) < 2:
        return 1.0

    predictions = np.array(predictions)
    consistency_score = 1.0 / (1.0 + np.std(predictions))

    return min(1.0, consistency_score)


def estimate_noise_level(model, data_loader):
    model.eval()
    residuals = []

    with torch.no_grad():
        for (seq, label) in data_loader:
            seq = seq.to(next(model.parameters()).device)
            label = label.to(next(model.parameters()).device)
            pred = model(seq)
            residual = torch.abs(pred - label)
            residuals.extend(residual.cpu().numpy())

    if len(residuals) == 0:
        return 0.0

    noise_level = np.mean(residuals)
    normalized_noise = 1.0 / (1.0 + noise_level)

    return min(1.0, normalized_noise)


def calculate_overall_quality_score(quality_metrics):
    volume_score = min(1.0, quality_metrics['data_volume'] / 1000)
    variance_score = min(1.0, quality_metrics['data_variance'])
    consistency_score = quality_metrics['prediction_consistency']
    noise_score = quality_metrics['noise_level']

    overall_score = (0.2 * volume_score + 0.3 * variance_score +
                     0.3 * consistency_score + 0.2 * noise_score)

    return overall_score


def personalize_model_for_geography(args, base_model, geographic_features):
    personalized_model = copy.deepcopy(base_model)

    if hasattr(personalized_model, 'adapt_personal_layers'):
        personalized_model.adapt_personal_layers(geographic_features)

    return personalized_model