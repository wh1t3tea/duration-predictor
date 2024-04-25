import torch


def masked_mse_loss(y_true, y_pred, mask):
    masked_mse_sum = torch.sum((y_pred - y_true) ** 2 * mask, dim=-1)
    non_padding_count = torch.sum(mask, dim=-1)
    masked_mse = masked_mse_sum / non_padding_count
    return torch.mean(masked_mse)


