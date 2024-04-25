import torch


def masked_mse_loss(y_true, y_pred, mask):
    masked_mse_sum = torch.sum((y_pred - y_true) ** 2 * mask, dim=-1)
    non_padding_count = torch.sum(mask, dim=-1)
    masked_mse = masked_mse_sum / non_padding_count
    return torch.mean(masked_mse)


def hard_mse_loss(y_true,
                  y_pred,
                  mask,
                  threshold):
    masked_mse = (y_pred - y_true) ** 2 * mask
    w_mse_ = []
    for row in masked_mse:
        row_list = []
        for x in row:
            if x <= threshold:
                row_list.append(x)
            else:
                row_list.append(x * max((x - threshold), 1))
        w_mse_.append(row_list)
    w_mse_sum = torch.sum(torch.tensor(w_mse_), dim=-1)
    non_padding_count = torch.sum(mask, dim=-1)
    masked_mse = w_mse_sum / non_padding_count
    return torch.mean(masked_mse)
