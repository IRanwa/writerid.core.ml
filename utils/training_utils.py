import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



def fit_episode(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device
) -> float:
    model.train()
    optimizer.zero_grad()

    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)

    if query_labels.numel() == 0 or query_images.numel() == 0:
        return 0.0
    classification_scores = model(support_images, support_labels, query_images)
    
    loss = criterion(classification_scores, query_labels)
    if torch.isnan(loss):
        return 0.0
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_on_one_task(
    model: nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    model.eval()

    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)

    with torch.no_grad():
        classification_scores = model(support_images, support_labels, query_images).detach()
    
    predicted_labels = torch.max(classification_scores, 1)[1]
    correct_count = (predicted_labels == query_labels).sum().item()
    total_count = len(query_labels)

    true_labels_np = query_labels.cpu().numpy()
    pred_labels_np = predicted_labels.cpu().numpy()
    return correct_count, total_count, true_labels_np, pred_labels_np

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_way_for_metrics: int
) -> Dict:
    total_predictions = 0
    correct_predictions = 0
    all_true_labels_for_epoch, all_pred_labels_for_epoch = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Validating"):
            if not batch_data or len(batch_data) != 5: continue
            support_images, support_labels, query_images, query_labels_task, _ = batch_data
            if support_images.numel() == 0 or query_images.numel() == 0: continue

            correct, total, true_labels, pred_labels = evaluate_on_one_task(
                model, support_images, support_labels, query_images, query_labels_task, device
            )
            correct_predictions += correct
            total_predictions += total
            all_true_labels_for_epoch.extend(true_labels)
            all_pred_labels_for_epoch.extend(pred_labels)

    return calculate_metrics(correct_predictions, total_predictions, all_true_labels_for_epoch, all_pred_labels_for_epoch, n_way_for_metrics, len(data_loader))

def evaluate_model_with_embeddings(data_loader: DataLoader, device: torch.device, n_way: int) -> Dict:
    """A separate evaluation function that works with pre-computed embeddings."""
    total_predictions = 0
    correct_predictions = 0
    all_true_labels_for_epoch, all_pred_labels_for_epoch = [], []
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating (Embeddings)"):
            if not batch_data or len(batch_data) != 5: continue
            support_embeddings, support_labels, query_embeddings, query_labels, _ = batch_data
            if support_embeddings.numel() == 0 or query_embeddings.numel() == 0: continue
            
            support_embeddings = support_embeddings.to(device)
            support_labels = support_labels.to(device)
            query_embeddings = query_embeddings.to(device)
            query_labels = query_labels.to(device)
            
            unique_labels = torch.unique(support_labels)
            prototypes = torch.stack([
                support_embeddings[torch.nonzero(support_labels == i).squeeze(-1)].mean(0)
                for i in unique_labels
            ])
            
            dists = torch.cdist(query_embeddings, prototypes)
            scores = -dists
            
            predicted_labels = torch.max(scores, 1)[1]
            correct_predictions += (predicted_labels == query_labels).sum().item()
            total_predictions += len(query_labels)
            all_true_labels_for_epoch.extend(query_labels.cpu().numpy())
            all_pred_labels_for_epoch.extend(predicted_labels.cpu().numpy())

    return calculate_metrics(correct_predictions, total_predictions, all_true_labels_for_epoch, all_pred_labels_for_epoch, n_way, len(data_loader))

def calculate_metrics(correct_preds, total_preds, true_labels, pred_labels, n_way, num_tasks):
    """Helper function to calculate and print metrics."""
    metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": None}
    if total_preds == 0:
        print("Evaluation completed, but no predictions were made.")
        return metrics

    metrics["accuracy"] = 100 * correct_preds / total_preds
    unique_task_labels = range(n_way)
    
    metrics["f1"] = f1_score(true_labels, pred_labels, average='macro', labels=unique_task_labels, zero_division=0) * 100
    metrics["precision"] = precision_score(true_labels, pred_labels, average='macro', labels=unique_task_labels, zero_division=0) * 100
    metrics["recall"] = recall_score(true_labels, pred_labels, average='macro', labels=unique_task_labels, zero_division=0) * 100
    metrics["confusion_matrix"] = confusion_matrix(true_labels, pred_labels, labels=unique_task_labels).tolist()
    
    print(
        f"Model tested on {num_tasks} tasks. "
        f"Accuracy: {metrics['accuracy']:.2f}% ({correct_preds}/{total_preds}), "
        f"F1: {metrics['f1']:.2f}%, Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%"
    )  
    return metrics

def sliding_average(value_list: List[float], window: int) -> float:
    if not value_list: return 0.0
    return np.asarray(value_list[-min(len(value_list), window):]).mean()



