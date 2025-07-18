import numpy as np
import torch


def evaluate_and_print_for_binary_classification(X, y, model):
    """
    Evaluates a PyTorch binary classification model on test data.

    Args:
        X (torch.Tensor): Test features, shape (num_samples, num_features)
        y (torch.Tensor): True labels, shape (num_samples,)
        model (torch.nn.Module): Trained PyTorch model

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = (outputs.squeeze() >= 0).long()
        correct = (preds == y).sum().item()
        accuracy = correct / y.shape[0]
        print(f"Accuracy: {accuracy * 100:.2f}%")
        for i in range(min(5, y.shape[0])):
            pred_prob = torch.sigmoid(outputs[i]).item()
            print(
                f"Sample {i}: True label = {y[i].item()}, Predicted = {preds[i].item()}, Probability = {pred_prob:.4f}")