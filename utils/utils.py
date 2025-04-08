import torch
import numpy as np
import pandas as pd
import random
import argparse
import os
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve

def print_metrics(name, metrics):
    if metrics is None:
        return    
    
    print(f"\n{name} Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.5f}" if isinstance(value, float) else f"{key}: {value}")

def save_metrics_to_csv(filename, metrics):
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def contact_metric(y_pred, y, threshold=0.2):
    y = y < 8.0
    #y_pred[y_pred > 8.0] = 10.0
    y_pred = torch.clip(1-(y_pred / 10), min=1e-6, max=0.99999)
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(y_pred, y.float())
    y = y.int()
    acc = torchmetrics.functional.accuracy(y_pred, y, threshold=threshold, task="binary")
    auroc = torchmetrics.functional.auroc(y_pred, y, task="binary")

    # Calculate precision, recall, and f1_score
    precision = torchmetrics.functional.precision(y_pred, y, threshold=threshold, average='none', task="binary")
    recall = torchmetrics.functional.recall(y_pred, y, threshold=threshold, average='none', task="binary")
    f1 = torchmetrics.functional.f1_score(y_pred, y, threshold=threshold, average='none', task="binary")
    return loss.cpu().tolist(), acc.cpu().tolist(), auroc.cpu().tolist(), precision.cpu().tolist(), recall.cpu().tolist(), f1.cpu().tolist()

def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred,y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))

import numpy as np
from scipy.stats import kendalltau

def kendall_tau(x, y):
    tau, _ = kendalltau(x, y)
    return tau

import numpy as np

def predictive_index(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    n = len(pred)
    ws, cs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(true[j] - true[i])
            c = -1
            if (pred[j] - pred[i]) * (true[j] - true[i]) > 0:
                c = 1
            elif true[j] - true[i] == 0:
                c = 0
            ws.append(w)
            cs.append(c)
    ws = np.array(ws)
    cs = np.array(cs)
    return np.sum(ws * cs) / np.sum(ws)

def affinity_metrics(affinity_pred, affinity):
    pearson = torchmetrics.functional.pearson_corrcoef(affinity_pred, affinity)
    rmse = torchmetrics.functional.mean_squared_error(affinity_pred, affinity, squared=False)
    spearman = torchmetrics.functional.spearman_corrcoef(affinity_pred, affinity)
    mae = torchmetrics.functional.mean_absolute_error(affinity_pred, affinity)
    sd = SD(affinity, affinity_pred)
    kt = kendall_tau(affinity_pred, affinity)
    pi = predictive_index(affinity_pred, affinity)
    
    return {
        "pearson": pearson.cpu().tolist(), 
        "SD": sd.cpu().tolist(),
        "rmse": rmse.cpu().tolist(), 
        "spearman": spearman.cpu().tolist(), 
        "KT": kt, 
        "PI": pi, 
        "mae": mae.cpu().tolist()
    }

import matplotlib.pyplot as plt
def plot_loss_curves(train_curve, val_curve, save_path):

    plt.figure(figsize=(10, 6))

    train_losses = [epoch_data['affinity_loss'] for epoch_data in train_curve]
    val_losses = [epoch_data['affinity_loss'] for epoch_data in val_curve]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curves.png'))
    plt.close()


